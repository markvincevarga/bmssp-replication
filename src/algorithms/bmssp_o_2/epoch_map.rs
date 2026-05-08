pub struct EpochMap {
    epochs: Vec<u64>,
    current: u64,
}

impl EpochMap {
    pub fn new(capacity: usize) -> Self {
        Self {
            epochs: vec![0; capacity],
            current: 1,
        }
    }

    pub fn contains(&self, idx: usize) -> bool {
        self.epochs[idx] == self.current
    }

    pub fn insert(&mut self, idx: usize) {
        self.epochs[idx] = self.current;
    }

    pub fn clear(&mut self) {
        self.current += 1;
        if self.current == 0 {
            self.epochs.fill(0);
            self.current = 1;
        }
    }
}
