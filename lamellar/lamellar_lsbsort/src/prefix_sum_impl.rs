// This code is based on code provided by Ryan Friese.
// Adding parallel prefix sum to Lamellar is on the TODO list.

use lamellar::active_messaging::prelude::*;
use lamellar::array::prelude::*;

#[AmData]
struct ApplyPePrefix {
    array: LocalLockArray<i64>,
    sum: i64,
}

#[am]
impl LamellarAM for ApplyPePrefix {
    async fn exec(&self) {
        self.array
            .write_local_chunks(1)
            .await
            .for_each(move |mut chunk| {
                for i in chunk.iter_mut() {
                    *i += self.sum;
                }
            })
            .await;
    }
}

pub fn inclusive_prefix_sum(array: &LocalLockArray<i64>, world: &LamellarWorld) {
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();

    //an array to hold the sum of elements on each pe
    let pe_sums = AtomicArray::<i64>::new(world.team(), num_pes, Distribution::Block).block();

    // println!("computing prefix sum of");
    // array.print();

    let chunk_size = array.num_elems_local() / world.num_threads_per_pe();
    let local_chunk_sums = array
        .write_local_chunks(chunk_size)
        .block()
        .map(|mut chunk| {
            let mut sum = 0;
            for i in chunk.iter_mut() {
                sum += *i;
                *i = sum;
            }
            sum
        })
        .collect::<Vec<_>>(Distribution::Block)
        .block();

    //calculate the local sum for each pe, and store it into local element of pe_sums
    pe_sums
        .local_data()
        .at(0)
        .store(local_chunk_sums.iter().sum::<i64>());

    //calculate the local prefix sums
    let _ = array
        .write_local_chunks(chunk_size)
        .block()
        .enumerate()
        .for_each(move |(i, mut chunk)| {
            let sum = local_chunk_sums[0..i].iter().sum::<i64>();
            for i in chunk.iter_mut() {
                *i += sum;
            }
        })
        .block(); //using a safe array we dont actually care if this finishes before we move on

    pe_sums.barrier(); // ensure all pes have writen to pe_sums

    // println!("temp  prefix sum array is");
    // array.print();
    /*println!("in prefix sum array is");
    array.print();
    println!("pe_sums is");
    pe_sums.print();*/

    //calculate the pe prefix sums to reduce communication we only do this on pe 0
    if my_pe == 0 {
        let mut sum = 0;
        for (pe, pe_sum) in pe_sums.onesided_iter().into_iter().enumerate() {
            // this following is a bit inefficient as have to send the indices for each batch_add but simple to use the array api
            // let mut  pe_indices = array.first_global_index_for_pe(pe).unwrap()..=array.last_global_index_for_pe(pe).unwrap();
            // let _ = array.batch_add(&mut pe_indices as &mut dyn Iterator<Item=i64>, sum).spawn();

            //alteratively we can do this with an AM with much less overhead
            //println!("pe {:?} applying sum {:?}", pe, sum);

            if sum != 0 {
                let _ = world
                    .exec_am_pe(
                        pe,
                        ApplyPePrefix {
                            array: array.clone(),
                            sum,
                        },
                    )
                    .spawn();
            }

            sum += pe_sum;
        }
    }
    world.wait_all();
    world.barrier();

    // println!("prefix sum is");
    // array.print();
}

pub fn exclusive_prefix_sum(array: &LocalLockArray<i64>, world: &LamellarWorld) {
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();

    //an array to hold the sum of elements on each pe
    let pe_sums = AtomicArray::<i64>::new(world.team(), num_pes, Distribution::Block).block();

    // println!("computing prefix sum of");
    // array.print();

    let chunk_size = array.num_elems_local() / world.num_threads_per_pe();
    let local_chunk_sums = array
        .write_local_chunks(chunk_size)
        .block()
        .map(|mut chunk| {
            let mut sum = 0;
            for i in chunk.iter_mut() {
                let temp = *i;
                *i = sum;
                (sum += temp);
            }
            sum
        })
        .collect::<Vec<_>>(Distribution::Block)
        .block();

    //calculate the local sum for each pe, and store it into local element of pe_sums
    pe_sums
        .local_data()
        .at(0)
        .store(local_chunk_sums.iter().sum::<i64>());

    //calculate the local prefix sums
    let _ = array
        .write_local_chunks(chunk_size)
        .block()
        .enumerate()
        .for_each(move |(i, mut chunk)| {
            let sum = local_chunk_sums[0..i].iter().sum::<i64>();

            for i in chunk.iter_mut() {
                let temp = sum;
                *i += temp;
            }
        })
        .spawn(); //using a safe array we dont actually care if this finishes before we move on

    pe_sums.barrier(); // ensure all pes have writen to pe_sums

    // println!("temp  prefix sum array is");
    // array.print();
    /*println!("pe_sums is");
    pe_sums.print();*/

    //calculate the pe prefix sums to reduce communication we only do this on pe 0
    if my_pe == 0 {
        let mut sum = 0;
        for (pe, pe_sum) in pe_sums.onesided_iter().into_iter().enumerate() {
            // this following is a bit inefficient as have to send the indices for each batch_add but simple to use the array api
            // let mut  pe_indices = array.first_global_index_for_pe(pe).unwrap()..=array.last_global_index_for_pe(pe).unwrap();
            // let _ = array.batch_add(&mut pe_indices as &mut dyn Iterator<Item=i64>, sum).spawn();

            //alteratively we can do this with an AM with much less overhead
            //println!("pe {:?} applying sum {:?}", pe, sum);

            if sum != 0 {
                let _ = world
                    .exec_am_pe(
                        pe,
                        ApplyPePrefix {
                            array: array.clone(),
                            sum,
                        },
                    )
                    .spawn();
            }

            sum += pe_sum;
        }
    }
    world.wait_all();
    world.barrier();

    // println!("prefix sum is");
    // array.print();
}
