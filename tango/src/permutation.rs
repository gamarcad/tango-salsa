use std::collections::HashMap;
use rand::prelude::SliceRandom;

pub struct Permutation {
    size: usize,
    mapping: HashMap<usize, usize>,
}

impl Permutation  {

    

    // Create a new permutation for a given size
    pub fn new(size: usize) -> Self {
        let mut mapping = HashMap::new();
        let mut indices: Vec<usize> = (0..size).collect();
        indices.shuffle(&mut rand::thread_rng());

        for (i, &index) in indices.iter().enumerate() {
            mapping.insert(i, index);
        }

        Permutation { size, mapping }
    }

    pub fn identity( size : usize ) -> Self {
        let mut mapping = HashMap::new();
        let mut indices: Vec<usize> = (0..size).collect();

        for (i, &index) in indices.iter().enumerate() {
            mapping.insert(i, index);
        }

        Permutation { size, mapping }
    }

    // Permutes the list according to the current permutation
    pub fn permute<'a, T>(&self, list: &'a Vec<T>) -> Vec<&'a T> {
        assert_eq!(list.len(), self.size, "List size must match permutation size");

        let mut result = Vec::with_capacity(self.size);
        for i in 0..self.size {
            result.push(list.get(self.mapping[&i]).unwrap());
        }

        result
    }

    // Inverts the current permutation
    pub fn invert(&self) -> Self {
        let mut inverted_mapping = HashMap::new();
        for (&key, &value) in &self.mapping {
            inverted_mapping.insert(value, key);
        }

        Permutation {
            size: self.size,
            mapping: inverted_mapping,
        }
    }
}
