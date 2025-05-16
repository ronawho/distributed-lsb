#![allow(non_snake_case)]
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::sync::Mutex;
use std::time::Instant;

use lamellar::active_messaging::prelude::*;
use lamellar::array::prelude::*;
use lamellar::array::Distribution;
use lamellar::darc::prelude::*;

const RADIX: u64 = 16; // TODO set to 16
const N_DIGITS: u64 = 64 / RADIX;
const N_BUCKETS: u64 = 1 << RADIX;
const MASK: u64 = N_BUCKETS - 1;

// An Active message based approach to LSB Radix Sort
// Each PE is responsible for N_BUCKETS/num_pes buckets
// data is sorted into buckets based on the current digit.
// At the end of the sort, data is evenly re-distributed across the pes.
// During the sort it is possible to become unbalanced
// based on the distribution of the keys. For uniform data
// this shouldnt be too bad.

#[lamellar::AmData(Default, Debug, ArrayOps, PartialEq, PartialOrd)]
struct SortElement {
    key: u64,
    val: u64,
}

// place local data from the original array into local buckets
#[AmLocalData]
struct InitAm {
    a: LocalLockArray<SortElement>,
    n_per_task: usize,
    tid: usize,
}

#[local_am]
impl InitAm {
    async fn exec(&self) -> Vec<Vec<SortElement>> {
        let start = self.tid * self.n_per_task;
        let end = start + self.n_per_task;
        let my_data = &self.a.read_local_data().await[start..end];
        let mut my_buckets = vec![vec![]; N_BUCKETS as usize];
        for ele in my_data {
            let i = get_bucket(ele, 0);
            my_buckets[i].push(*ele);
        }
        my_buckets
    }
}

// update my portion of the global buckets with data from the other pes
#[AmData]
struct BucketAm {
    buckets: Darc<Vec<Vec<Mutex<Vec<SortElement>>>>>, //bucket, pe, elements
    data: Vec<Vec<SortElement>>,
    n_buckets_per_pe: usize,
    orig_pe: usize,
}

#[am]
impl BucketAm {
    async fn exec(&self) {
        let start = lamellar::current_pe * self.n_buckets_per_pe;
        let end = start + self.n_buckets_per_pe;
        for (bucket, d) in self.buckets[start..end].iter().zip(self.data.iter()) {
            bucket[self.orig_pe].lock().unwrap().extend_from_slice(d);
        }
    }
}

// read the data in my portion of the global buckets and sort into local buckets based on current digit
#[AmLocalData]
struct LocalBucketAm {
    buckets: Darc<Vec<Vec<Mutex<Vec<SortElement>>>>>, //bucket, pe, elements
    n_buckets_per_pe: usize,
    n_buckets_per_task: usize,
    tid: usize,
    digit: u64,
}

#[local_am]
impl LocalBucketAm {
    async fn exec(&self) -> Vec<Vec<SortElement>> {
        let pe_start = lamellar::current_pe * self.n_buckets_per_pe;
        let thread_start = pe_start + self.tid * self.n_buckets_per_task;
        let thread_end = thread_start + self.n_buckets_per_task;
        let mut my_buckets = vec![vec![]; N_BUCKETS as usize];
        for buckets in &self.buckets[thread_start..thread_end] {
            for pe_b in buckets.iter() {
                let mut pe_b = pe_b.lock().unwrap();
                for ele in pe_b.drain(..) {
                    let i = get_bucket(&ele, self.digit);
                    my_buckets[i].push(ele);
                }
            }
        }
        my_buckets
    }
}

// take the data from the global buckets and put it into the local array
// this ensures that the data is evenly distributed across the pes
#[AmData]
struct RedistributeAm {
    buckets: Darc<Vec<Vec<Mutex<Vec<SortElement>>>>>, //bucket, pe, elements
    array: LocalLockArray<SortElement>,
    n_buckets_per_pe: usize,
    n_buckets_per_task: usize,
    pe_start: usize,
}

#[am]
impl RedistributeAm {
    async fn exec(&self) {
        let tasks = (0..lamellar::world.num_threads_per_pe())
            .map(|tid| {
                lamellar::world
                    .exec_am_local(FlattenDataAm {
                        buckets: self.buckets.clone(),
                        n_buckets_per_pe: self.n_buckets_per_pe,
                        n_buckets_per_task: self.n_buckets_per_task,
                        tid,
                    })
                    .spawn()
            })
            .collect::<Vec<_>>();
        let mut flat_data = vec![];
        for task in tasks {
            flat_data.extend_from_slice(&task.await);
        }

        // this isnt really unsafe as we are protecting simultaneous writes to array via the LocalLockArrayType
        // and likely should be a change in the API
        unsafe { self.array.put(self.pe_start, flat_data).await };
    }
}

// take our portion of the global buckets and flatten it into a single vector
#[AmLocalData]
struct FlattenDataAm {
    buckets: Darc<Vec<Vec<Mutex<Vec<SortElement>>>>>,
    n_buckets_per_pe: usize,
    n_buckets_per_task: usize,
    tid: usize,
}

#[local_am]
impl FlattenDataAm {
    async fn exec(&self) -> Vec<SortElement> {
        let pe_start = lamellar::current_pe * self.n_buckets_per_pe;
        let thread_start = pe_start + self.tid * self.n_buckets_per_task;
        let thread_end = thread_start + self.n_buckets_per_task;
        let mut flat_data = vec![];
        for buckets in &self.buckets[thread_start..thread_end] {
            for pe_b in buckets.iter() {
                let mut pe_b = pe_b.lock().unwrap();
                flat_data.append(&mut pe_b.drain(..).collect());
            }
        }
        flat_data
    }
}

fn get_bucket(x: &SortElement, d: u64) -> usize {
    return ((x.key >> (RADIX * d)) & MASK) as usize;
}

fn global_shuffle(A: &LocalLockArray<SortElement>, world: &LamellarWorld, n_per_task: usize) {
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let n_tasks_per_pe = world.num_threads_per_pe();

    let mut buckets: Vec<Vec<Mutex<Vec<SortElement>>>> = vec![];
    for _ in 0..N_BUCKETS {
        let mut b = vec![];
        for _ in 0..num_pes {
            b.push(Mutex::new(vec![]));
        }
        buckets.push(b);
    }
    let buckets = Darc::new(world, buckets).block().unwrap();

    let n_buckets_per_pe = N_BUCKETS as usize / num_pes;

    let counts_f = AtomicArray::new(world, num_pes, Distribution::Block).spawn(); // well need this later, might as well let it initialize while we do other work
    for digit in 0..N_DIGITS {
        // sort local data into buckets based on current digit
        let mut thread_buckets = (0..n_tasks_per_pe)
            .map(|tid| {
                if digit == 0 {
                    world
                        .exec_am_local(InitAm {
                            a: A.clone(),
                            n_per_task,
                            tid,
                        })
                        .spawn()
                } else {
                    world
                        .exec_am_local(LocalBucketAm {
                            buckets: buckets.clone(),
                            n_buckets_per_pe,
                            n_buckets_per_task: n_buckets_per_pe / n_tasks_per_pe,
                            tid,
                            digit,
                        })
                        .spawn()
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|task| task.block())
            .collect::<Vec<_>>();

        //reduce the thread buckets into pe buckets
        let mut pe_buckets = vec![vec![]; N_BUCKETS as usize];
        for (i, b) in pe_buckets.iter_mut().enumerate() {
            for tb in thread_buckets.iter_mut() {
                b.extend_from_slice(&tb[i]);
                tb[i].clear();
            }
        }
        world.barrier();
        // send my buckets to the appropriate PEs
        for pe in (0..num_pes).rev() {
            let data = pe_buckets.split_off(pe_buckets.len() - n_buckets_per_pe);
            let _ = world
                .exec_am_pe(
                    pe,
                    BucketAm {
                        buckets: buckets.clone(),
                        data,
                        n_buckets_per_pe,
                        orig_pe: my_pe,
                    },
                )
                .spawn();
        }
        world.wait_all();
        world.barrier();
    }

    //Evenly distributed the final data across the pes
    // get the number of elements on each pe
    let mut cnt = 0;
    for bucket in buckets[my_pe * n_buckets_per_pe..(my_pe + 1) * n_buckets_per_pe].iter() {
        for pe_b in bucket {
            let pe_b = pe_b.lock().unwrap();
            cnt += pe_b.len();
        }
    }
    let counts = counts_f.block();
    counts.local_data().at(0).store(cnt);
    counts.barrier();
    if my_pe == 0 {
        let mut cur_start = 0;
        for (pe, pe_count) in counts.onesided_iter().into_iter().enumerate() {
            let _ = world
                .exec_am_pe(
                    pe,
                    RedistributeAm {
                        buckets: buckets.clone(),
                        array: A.clone(),
                        n_buckets_per_pe,
                        n_buckets_per_task: n_buckets_per_pe / n_tasks_per_pe,
                        pe_start: cur_start,
                    },
                )
                .spawn();
            cur_start += pe_count;
        }
    }

    world.wait_all();
    world.barrier();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let world = lamellar::LamellarWorldBuilder::new().build();
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let mut n: usize = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| 64);
    // TODO: try multiplying n_tasks_per_pe by 2
    let n_tasks_per_pe: usize = world.num_threads_per_pe();
    let n_per_task: usize = usize::div_ceil(n, num_pes * n_tasks_per_pe);
    let n_per_pe: usize = n_tasks_per_pe * n_per_task;
    let n_tasks_total = num_pes * n_tasks_per_pe;
    n = n_per_task * n_tasks_total;
    if my_pe == 0 {
        println!("hello from pe {} of {}", my_pe, num_pes);
        println!("there are {} tasks per pe", n_tasks_per_pe);
        println!(
            "n is {} n_per_pe is {} n_per_task is {}",
            n, n_per_pe, n_per_task
        );
    }

    let A = LocalLockArray::<SortElement>::new(world.team(), n, Distribution::Block).block();

    let time_rng = Instant::now();
    let glob_start = n_per_pe * my_pe;
    A.write_local_chunks(n_per_task)
        .block()
        .enumerate()
        .for_each(move |(tid, mut task_slice)| {
            //println!("tid {:?} got chunk {:?}", tid, task_slice);
            let seed = (42 + my_pe * n_tasks_per_pe + tid) as u64;
            let mut rng = Pcg64::seed_from_u64(seed);
            for (i, elt) in task_slice.iter_mut().enumerate() {
                let idx = (glob_start + tid * n_per_task + i) as u64;
                let r = rng.sample(StandardUniform);
                *elt = SortElement { key: r, val: idx };
            }
            //println!("tid {:?} ->  chunk {:?}", tid, task_slice);
        })
        .block();

    world.barrier();

    if my_pe == 0 {
        println!("elapsed rng time {}", time_rng.elapsed().as_secs_f64());
    }
    world.barrier();

    let time_sort = Instant::now();
    global_shuffle(&A, &world, n_per_task);
    world.barrier();

    if my_pe == 0 {
        let time_sort = time_sort.elapsed().as_secs_f64();
        println!(
            "elapsed sort time {} M elems/s = {}",
            time_sort,
            (n as f64 / 1000000.0) / time_sort
        );
    }

    //roughly check that the data is sorted
    A.read_local_chunks(n_per_task)
        .block()
        .for_each(move |task_slice| {
            let mut failed = vec![];
            for i in 1..task_slice.len() {
                let prev = task_slice[i - 1].key;
                let curr = task_slice[i].key;
                if prev > curr {
                    failed.push((i, prev, curr));
                }
            }
            if failed.len() > 0 {
                println!(" [{}]  {:?}: ", my_pe, failed);
            }
        })
        .block();
}
