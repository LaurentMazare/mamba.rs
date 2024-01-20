use anyhow::Result;

pub struct Tokenizer {
    tokens: Vec<String>,
}

impl Tokenizer {
    pub fn from_vocab_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let content = std::fs::read_to_string(p)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;
        let config = match config.as_object() {
            Some(config) => config,
            None => anyhow::bail!("not an object"),
        };
        let mut tokens = vec![];
        for (key, value) in config.iter() {
            let value = match value.as_u64() {
                Some(value) => value as usize,
                None => anyhow::bail!("value attached to {key} is not an int {value:?}"),
            };
            if tokens.len() <= value {
                tokens.resize_with(value + 1, String::new);
            }
            tokens[value] = key.to_string()
        }
        Ok(Self { tokens })
    }

    pub fn tokens(&self, idx: usize) -> Result<&str> {
        if idx >= self.tokens.len() {
            anyhow::bail!("token out of bounds {idx} >= {}", self.tokens.len())
        }
        Ok(self.tokens[idx].as_str())
    }
}
