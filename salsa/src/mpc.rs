use rand::Rng;




fn random_bit() -> u8 {
    let mut rng = rand::thread_rng();
    let r_boolean : bool = rng.gen();
    return r_boolean as u8;
}
pub(crate) fn share(a: u8) -> (u8, u8) {
    let r: u8 = random_bit();
    let s0 = r;
    let s1 = a ^ r;
    (s0, s1)
}

pub(crate) fn share_arithmetic(a : u64) -> (i64, i64) {
    let mut rng = rand::thread_rng();
    let r : u32 = rng.gen();
    return (r as i64, a as i64 - r as i64);
}

pub(crate) fn add_arithmetic( a : (i64, i64), b : (i64, i64)) -> (i64, i64) {
    ( a.0 + b.0, a.1 + b.1 )
}
pub(crate) fn reconstruct_arithmetic(a : (i64, i64) ) -> u64 {
    (a.0 + a.1) as u64
}

pub(crate) fn xor_gate(a: (u8, u8), b: (u8, u8)) -> (u8, u8) {
    let (a0, a1) = a;
    let (b0, b1) = b;
    let c0 = a0 ^ b0;
    let c1 = a1 ^ b1;
    (c0, c1)
}

pub(crate) fn and_gate(a: (u8, u8), b: (u8, u8)) -> (u8, u8) {
    // generate a beaver triplet (x, y, z = xy)
    let x: u8 = random_bit();
    let y: u8 = random_bit();
    let z: u8 = x & y;
    let (x0, x1) = share(x);
    let (y0, y1) = share(y);
    let (z0, z1) = share(z);

    let (a0, a1) = a;
    let (b0, b1) = b;

    // compute e = a ^ x, f = b ^ y
    let i0 = a0 ^ x0;
    let i1 = a1 ^ x1;
    let j0 = b0 ^ y0;
    let j1 = b1 ^ y1;

    // reconstruct i and j
    let e = i0 ^ i1;
    let f = j0 ^ j1;

    // compute c0 and c1
    let c0 = 0 * e * f ^ f * x0 ^ e * y0 ^ z0;
    let c1 = 1 * e * f ^ f * x1 ^ e * y1 ^ z1;

    (c0, c1)
}


pub(crate) fn or_gate(a: (u8, u8), b: (u8, u8)) -> (u8, u8) {
    let tmp1 = xor_gate(a, b);
    let tmp2 = and_gate(a, b);
    xor_gate(tmp1, tmp2)
}

pub(crate) fn reconstruct(s: (u8, u8)) -> u8 {
    let (s0, s1) = s;
    s0 ^ s1
}

pub(crate) fn share_u64(v: u64) -> Vec<(u8, u8)> {
    let mut bits = Vec::with_capacity(64);
    let mut value = v;
    for _ in 0..64 {
        bits.push(share((value & 1) as u8));
        value >>= 1;
    }
    bits
}

pub(crate) fn lt_u64(a: Vec<(u8, u8)>, b: Vec<(u8, u8)>) -> (u8, u8) {
    let mut a_geq_b = share(1);
    let mut free = share(1);

    for i in (0..64).rev() {
        let diff = xor_gate(a[i as usize], b[i as usize]);
        let ai_gt_bi = and_gate(diff, b[i as usize]);

        let c = and_gate(free, diff);
        let not_c = xor_gate(c, share(1));
        a_geq_b = or_gate(
            and_gate(c, xor_gate(ai_gt_bi, share(1))),
            and_gate(not_c, a_geq_b),
        );
        free = or_gate(
            and_gate(c, xor_gate(free, share(1))),
            and_gate(not_c, free),
        );
    }

    xor_gate(a_geq_b, share(1))
}

pub(crate) fn eq_u64(a: Vec<(u8, u8)>, b: Vec<(u8, u8)>) -> (u8, u8) {
    let diffs: Vec<(u8, u8)> = a.iter().zip(b.iter()).map(|(&ai, &bi)| xor_gate(ai, bi)).collect();
    let mut result = or_gate(diffs[0], diffs[1]);
    for i in 2..64 {
        result = or_gate(result, diffs[i as usize]);
    }
    xor_gate(result, share(1))
}

pub(crate) fn leq_u64(a: Vec<(u8, u8)>, b: Vec<(u8, u8)>) -> (u8, u8) {
    let a_lt_b = lt_u64(a.clone(), b.clone());
    let a_eq_b = eq_u64(a, b);
    or_gate(a_lt_b, a_eq_b)
}

pub(crate) fn if_then_else_u64(c: (u8, u8), a: Vec<(u8, u8)>, b: Vec<(u8, u8)>) -> Vec<(u8, u8)> {
    let not_c = xor_gate(c, share(1));
    let mut result = Vec::with_capacity(64);

    for i in 0..64 {
        result.push(or_gate(
            and_gate(c, a[i as usize]),
            and_gate(not_c, b[i as usize]),
        ));
    }

    result
}

pub(crate) fn argmax(values: Vec<Vec<(u8, u8)>>) -> Vec<(u8, u8)> {
    let mut best_index = share_u64(0);
    let mut best_score = values.first().unwrap().clone();

    for (index, score) in values.iter().enumerate() {
        let best_less_than_score = lt_u64(best_score.clone(), score.clone());
        best_index = if_then_else_u64(best_less_than_score.clone(), share_u64(index as u64), best_index);
        best_score = if_then_else_u64(best_less_than_score.clone(), score.clone(), best_score.clone());
    }

    best_index
}

pub(crate) fn selection_bits(best_index: Vec<(u8, u8)>, k: usize) -> Vec<(u8, u8)> {
    (0..k).map(|index| eq_u64(best_index.clone(), share_u64(index as u64))).collect()
}

pub(crate) fn reconstruct_u64(bits: Vec<(u8, u8)>) -> u64 {
    let reconstructed_bits: Vec<u8> = bits.iter().map(|&s| reconstruct(s)).collect();
    reconstructed_bits.iter().enumerate().fold(0, |acc, (i, &b)| acc | (u64::from(b) << i))
}