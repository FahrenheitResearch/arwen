//! Replay the complete pristine MPAS container APIs against Rust's containers.
//! These semantic equivalents are used by the static mapper/interpolator.
//! MPAS: mpas_init_atm_bitarray.F:30-213; mpas_init_atm_queue.F:44-245;
//! mpas_stack.F:121-258. Native empty-queue return words are undefined;
//! their logging branches are exercised without observing undefined return words.
use std::{cell::RefCell, collections::VecDeque, rc::Rc};

#[test]
fn static_collections_match_complete_pristine_modules() {
    let mut words = Vec::<u32>::new();
    fn dump_bits(words: &mut Vec<u32>, bits: &[bool], nx: usize, ny: usize) {
        let width = (nx as f32 / 32.0).ceil() as usize;
        words.extend([nx as u32, ny as u32, width as u32, ny as u32, 32]);
        for j in 0..ny {
            for block in 0..width {
                let mut value = 0u32;
                for bit in 0..32 {
                    let i = block * 32 + bit;
                    if i < nx && bits[j * nx + i] { value |= 1u32 << bit; }
                }
                words.push(value);
            }
        }
    }
    for nx in [0, 1, 31, 32, 33, 63, 64, 97] {
        for ny in 0..=4 {
            let mut a = vec![false; nx * ny];
            dump_bits(&mut words, &a, nx, ny);
            for j in 1..=ny { for i in 1..=nx {
                if (i * 7 + j * 3) % 5 < 2 { a[(j - 1) * nx + i - 1] = true; }
            }}
            let b = a.clone();
            for j in 1..=ny { for i in 1..=nx {
                if (i * 3 + j * 7) % 7 == 0 { a[(j - 1) * nx + i - 1] = false; }
            }}
            dump_bits(&mut words, &a, nx, ny);
            dump_bits(&mut words, &b, nx, ny);
            let mut c = vec![false; nx * ny];
            for j in 1..=ny { for i in 1..=nx {
                if (i + j) % 3 == 0 { c[(j - 1) * nx + i - 1] = true; }
            }}
            for (value, other) in a.iter_mut().zip(c) { *value |= other; }
            dump_bits(&mut words, &a, nx, ny);
            words.extend(a.into_iter().map(u32::from));
        }
    }
    for n in 0i32..=64 {
        let mut queue = VecDeque::<[u32; 4]>::new();
        words.extend([u32::from(!queue.is_empty()), queue.len() as u32]);
        for i in 1..=n {
            queue.push_back([0x80000000u32 + i as u32, 1 + i as u32 * 12345,
                             (i * 17 - 99) as u32, ((i * 23) % 71 - 35) as u32]);
            words.extend(*queue.front().unwrap());
            words.push(queue.len() as u32);
        }
        for _ in 1..=n/2 {
            words.extend(queue.pop_front().unwrap());
            words.extend([queue.len() as u32, u32::from(!queue.is_empty())]);
        }
        queue.clear();
        words.push(queue.len() as u32);
        for i in 1..=n {
            queue.push_back([(i as f32).to_bits(), (-i as f32).to_bits(), i as u32, (-i) as u32]);
        }
        while let Some(item) = queue.pop_front() { words.extend(item); }
        words.push(u32::from(!queue.is_empty()));
    }
    for n in 0usize..=64 {
        let mut stack = Vec::<Rc<RefCell<i32>>>::new();
        words.push(u32::from(stack.is_empty()));
        words.push(u32::from(stack.pop().is_some()));
        let shared: Vec<_> = (1..=8).map(|i| Rc::new(RefCell::new(i * 11))).collect();
        for i in 0..n { stack.push(Rc::clone(&shared[i % 8])); }
        for item in &shared { *item.borrow_mut() += 1000; }
        words.push(u32::from(stack.is_empty()));
        for _ in 0..n/2 { words.push(*stack.pop().unwrap().borrow() as u32); }
        stack.clear();
        words.push(u32::from(stack.is_empty()));
        words.extend(shared.iter().map(|item| *item.borrow() as u32));
        for _ in 0..2 {
            for i in 1..=n { stack.push(Rc::new(RefCell::new(i as i32))); }
            stack.clear();
            words.push(u32::from(stack.is_empty()));
        }
    }
    let expected: Vec<u32> = include_str!("native_static/collections.words").lines()
        .map(|line| u32::from_str_radix(line, 16).unwrap()).collect();
    assert_eq!(words.len(), expected.len());
    let differences: Vec<_> = words.iter().zip(&expected).enumerate()
        .filter(|(_, (a, b))| a != b).take(10).collect();
    assert!(differences.is_empty(), "native container words differ: {differences:?}");
    println!("native container words={}, differing=0", words.len());
}
