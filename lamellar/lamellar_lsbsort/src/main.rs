use rayon::prelude::*;
use rand::prelude::*;
use rand::distr::StandardUniform;
use rand_pcg::Pcg64;
use std::time::Instant;

use lamellar::array::prelude::*;
use lamellar::array::Distribution;

const RADIX: u64 = 16;
const N_DIGITS: u64 = 64 / RADIX;
const N_BUCKETS: u64 = 1 << RADIX;
const COUNTS_SIZE: u64 = N_BUCKETS;
const MASK: u64 = N_BUCKETS - 1;

#[lamellar::AmData(
    Default,
    Debug,
    ArrayOps(Arithmetic, CompExEps, Shift),
    PartialEq,
    PartialOrd
)]
struct SortElement {
    key: u64,
    val: u64,
}

impl std::ops::AddAssign for SortElement {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            key: self.key + other.key,
            val: self.val + other.val,
        }
    }
}

impl std::ops::SubAssign for SortElement {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            key: self.key - other.key,
            val: self.val - other.val,
        }
    }
}

impl std::ops::Sub for SortElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            key: self.key - other.key,
            val: self.val - other.val,
        }
    }
}

impl std::ops::MulAssign for SortElement {
    fn mul_assign(&mut self, other: Self) {
        *self = Self {
            key: self.key * other.key,
            val: self.val * other.val,
        }
    }
}

impl std::ops::DivAssign for SortElement {
    fn div_assign(&mut self, other: Self) {
        *self = Self {
            key: self.key / other.key,
            val: self.val / other.val,
        }
    }
}

impl std::ops::RemAssign for SortElement {
    fn rem_assign(&mut self, other: Self) {
        *self = Self {
            key: self.key % other.key,
            val: self.val % other.val,
        }
    }
}

impl std::ops::ShlAssign for SortElement {
    fn shl_assign(&mut self, other: Self) {
        self.key <<= other.key;
        self.val <<= other.val;
    }
}

impl std::ops::ShrAssign for SortElement {
    fn shr_assign(&mut self, other: Self) {
        self.key >>= other.key;
        self.val >>= other.val;
    }
}

fn get_bucket(x: SortElement, d: u64) -> usize {
    return ((x.key >> (RADIX*d)) & MASK) as usize;
}

/*fn local_shuffle(Src: &[SortElement]
                 Dst: &mut [SortElement],
                 starts: &mut [i64],
                 counts: &mut [i64],
                 digit: u64,
                 n: usize) {
    assert!(starts.len() == COUNTS_SIZE);
    assert!(counts.len() == COUNTS_SIZE);
    assert!(Src.len() == Dst.len());

    // clear out starts and counts
    starts.fill(0);
    counts.fill(0);

    // compute the count for each digit
    for elt in Src {
        counts[get_bucket(elt, digit)] += 1;
    }

    // compute the starts with an exclusive scan
    {
        let sum: i64 = 0;
        for (count, start) in zip(counts, starts) {
            start = sum;
            sum += count;
        }
    }

    // shuffle the data
    for elt in Src {
        let next_ptr: &mut i64 = starts[get_bucket(elt, digit)];
        Dst[next_ptr] = elt;
        next_ptr += 1;
    }
}*/

fn calc_global_index(bucket: usize, my_pe: usize, my_task: usize,
                     num_pes: usize, n_tasks: usize) {
    return ((bucket*num_pes*n_tasks) + (my_pe*n_tasks) + task);
}

// shuffle elements from A into B
fn global_shuffle(A: &UnsafeArray::<SortElement>,
                  B: &UnsafeArray::<SortElement>,
                  digit: u64,
                  counts: &UnsafeArray::<i64>,
                  starts: &UnsafeArray::<i64>,
                  world: &LamellarWorld,
                  n_per_task: usize) {

    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let n_tasks = lamellar::config().threads;
    let n_per_pe = n_tasks * n_per_task;
    let n = n_per_task * num_pes * n_tasks;

    let loc_A = A.mut_local_data();
    let loc_B = B.mut_local_data();

    // count the digits in each task's region
    let task_counts = vec![vec![0; COUNTS_SIZE]; n_tasks];

    rayon::scope(|s| {
        for tid in 0..n_tasks {
            s.spawn(move |_| {
                let mut mycounts = vec![0; COUNTS_SIZE];
                let start = tid * n_per_task;
                let end = start + n_per_task;
                for i in start..end {
                    loc_A[i]
        
                        counts[get_bucket(elt, digit)] += 1;

                }
            }
        }
    }

    A.par_chunks_mut
    let mut starts = vec![0; COUNTS_SIZE]; 
    let mut starts = vec![0; COUNTS_SIZE]; 

    });
}

// Sort the data in A, using B as scratch space.
fn my_sort(A: &UnsafeArray::<SortElement>,
           B: &UnsafeArray::<SortElement>,
           counts: &UnsafeArray::<i64>,
           starts: &UnsafeArray::<i64>,
           world: &LamellarWorld,
           n_per_task: usize) {

    assert!(N_DIGITS % 2 == 0);

    for digit in (0..N_DIGITS).step_by(2) {
        global_shuffle(A, B, digit, counts, starts, world, n_per_task);
        global_shuffle(B, A, digit+1, counts, starts, world, n_per_task);
    }
}

// TODO: batch_load, batch_store
// rayon scope / spawn
// par_chunks_mut
//
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let world = lamellar::LamellarWorldBuilder::new().build();
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let mut n = args.get(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| 100);
    let n_tasks = world.num_threads_per_pe()
    let n_per_task = usize::div_ceil(n, num_pes * n_tasks);
    let n_per_pe = n_tasks * n_per_task;
    n = n_per_task * num_pes * n_tasks;
    if my_pe == 0 {
        println!("hello from pe {} of {}", my_pe, num_pes);
        println!("n is {} n_per_pe is {} n_per_task is {}",
                 n, n_per_pe, n_per_task);
    }

    let A_f = UnsafeArray::<SortElement>::new(world.team(), n,
                                            Distribution::Block);
    let B_f = UnsafeArray::<SortElement>::new(world.team(), n,
                                            Distribution::Block);
    let starts_f = UnsafeArray::<i64>::new(world.team(),
                                           COUNTS_SIZE*n_tasks*num_pes,
                                           Distribution::Block);
    let counts_f = UnsafeArray::<i64>::new(world.team(),
                                           COUNTS_SIZE*n_tasks*num_pes,
                                           Distribution::Block);
    let A = A_f.block();
    let B = B_f.block();
    let starts = starts_f.block();
    let counts = counts_f.block();

    let time_rng = Instant::now();

    // set the input
    unsafe {
        /*input_array
            .dist_iter_mut()
            .enumerate()
            .map_with(rng, |rng, | rng.sample(Standard))
            .for_each(move |elem| *elem = (rand, idx));*/

        let rng = Pcg64::seed_from_u64((42 + my_pe) as u64);
        let start = (n_per_pe * my_pe) as u64;
        println!("start is {}", start);
        A.mut_local_data()
         .par_iter_mut()
         .enumerate()
         .map_with(rng, |rng, (i,elt)|
                   (elt, rng.sample(StandardUniform), i))
         .for_each(move|(elt, r, i)|
                   *elt = SortElement {key:r,
                                       val: (i as u64) + start});
    }

    world.barrier();

    println!("elapsed rng time {}", time_rng.elapsed().as_secs_f64());


    input_array.print();

    world.barrier();
    let time_sort = Instant::now();

    my_sort(A, B, counts, starts, world, n_per_pe, n_per_task);
    println!("elapsed sort time {}", time_rng.elapsed().as_secs_f64());

    world.barrier();
        /*
    let mut rng: StdRng = SeedableRng::seed_from_u64(my_pe as u64);

    // initialize arrays
    let array_init = unsafe {unsafe_array
        .dist_iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i)};
    // rand_index.dist_iter_mut().for_each(move |x| *x = rng.lock().gen_range(0,global_count)).wait(); //this is slow because of the lock on the rng so we will do unsafe slice version instead...
    unsafe {
        for elem in rand_index.as_mut_slice().unwrap().iter_mut() {
            *elem = rng.gen_range(0, global_count);
        }
    }
    world.block_on(array_init);
    let array = unsafe_array.into_read_only();
    // let rand_index = rand_index.into_read_only();
    world.barrier();

    if my_pe == 0 {
        println!("starting index gather");
    }

    let now = Instant::now();
    index_gather(&array, rand_index);

    if my_pe == 0 {
        println!("{:?} issue time {:?} ", my_pe, now.elapsed());
    }
    array.wait_all();
    if my_pe == 0 {
        println!(
            "local run time {:?} local mups: {:?}",
            now.elapsed(),
            (l_num_updates as f32 / 1_000_000.0) / now.elapsed().as_secs_f32()
        );
    }
    array.barrier();
    let global_time = now.elapsed().as_secs_f64();
    if my_pe == 0 {
        println!(
            "global time {:?} MB {:?} MB/s: {:?}",
            global_time,
            (world.MB_sent()),
            (world.MB_sent()) / global_time,
        );
        println!(
            "MUPS: {:?}",
            ((l_num_updates * num_pes) as f64 / 1_000_000.0) / global_time,
        );
        println!(
            "Secs: {:?}",
             global_time,
        );
        println!(
            "GB/s Injection rate: {:?}",
            (8.0 * (l_num_updates * 2) as f64 * 1.0E-9) / global_time,
        );
    }*/
}

