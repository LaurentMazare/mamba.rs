use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

const BYTES_TO_UNICODE: [(u8, char); 256] = [
    (33, '!'),
    (34, '"'),
    (35, '#'),
    (36, '$'),
    (37, '%'),
    (38, '&'),
    (39, '\''),
    (40, '('),
    (41, ')'),
    (42, '*'),
    (43, '+'),
    (44, ','),
    (45, '-'),
    (46, '.'),
    (47, '/'),
    (48, '0'),
    (49, '1'),
    (50, '2'),
    (51, '3'),
    (52, '4'),
    (53, '5'),
    (54, '6'),
    (55, '7'),
    (56, '8'),
    (57, '9'),
    (58, ':'),
    (59, ';'),
    (60, '<'),
    (61, '='),
    (62, '>'),
    (63, '?'),
    (64, '@'),
    (65, 'A'),
    (66, 'B'),
    (67, 'C'),
    (68, 'D'),
    (69, 'E'),
    (70, 'F'),
    (71, 'G'),
    (72, 'H'),
    (73, 'I'),
    (74, 'J'),
    (75, 'K'),
    (76, 'L'),
    (77, 'M'),
    (78, 'N'),
    (79, 'O'),
    (80, 'P'),
    (81, 'Q'),
    (82, 'R'),
    (83, 'S'),
    (84, 'T'),
    (85, 'U'),
    (86, 'V'),
    (87, 'W'),
    (88, 'X'),
    (89, 'Y'),
    (90, 'Z'),
    (91, '['),
    (92, '\\'),
    (93, ']'),
    (94, '^'),
    (95, '_'),
    (96, '`'),
    (97, 'a'),
    (98, 'b'),
    (99, 'c'),
    (100, 'd'),
    (101, 'e'),
    (102, 'f'),
    (103, 'g'),
    (104, 'h'),
    (105, 'i'),
    (106, 'j'),
    (107, 'k'),
    (108, 'l'),
    (109, 'm'),
    (110, 'n'),
    (111, 'o'),
    (112, 'p'),
    (113, 'q'),
    (114, 'r'),
    (115, 's'),
    (116, 't'),
    (117, 'u'),
    (118, 'v'),
    (119, 'w'),
    (120, 'x'),
    (121, 'y'),
    (122, 'z'),
    (123, '{'),
    (124, '|'),
    (125, '}'),
    (126, '~'),
    (161, '¡'),
    (162, '¢'),
    (163, '£'),
    (164, '¤'),
    (165, '¥'),
    (166, '¦'),
    (167, '§'),
    (168, '¨'),
    (169, '©'),
    (170, 'ª'),
    (171, '«'),
    (172, '¬'),
    (174, '®'),
    (175, '¯'),
    (176, '°'),
    (177, '±'),
    (178, '²'),
    (179, '³'),
    (180, '´'),
    (181, 'µ'),
    (182, '¶'),
    (183, '·'),
    (184, '¸'),
    (185, '¹'),
    (186, 'º'),
    (187, '»'),
    (188, '¼'),
    (189, '½'),
    (190, '¾'),
    (191, '¿'),
    (192, 'À'),
    (193, 'Á'),
    (194, 'Â'),
    (195, 'Ã'),
    (196, 'Ä'),
    (197, 'Å'),
    (198, 'Æ'),
    (199, 'Ç'),
    (200, 'È'),
    (201, 'É'),
    (202, 'Ê'),
    (203, 'Ë'),
    (204, 'Ì'),
    (205, 'Í'),
    (206, 'Î'),
    (207, 'Ï'),
    (208, 'Ð'),
    (209, 'Ñ'),
    (210, 'Ò'),
    (211, 'Ó'),
    (212, 'Ô'),
    (213, 'Õ'),
    (214, 'Ö'),
    (215, '×'),
    (216, 'Ø'),
    (217, 'Ù'),
    (218, 'Ú'),
    (219, 'Û'),
    (220, 'Ü'),
    (221, 'Ý'),
    (222, 'Þ'),
    (223, 'ß'),
    (224, 'à'),
    (225, 'á'),
    (226, 'â'),
    (227, 'ã'),
    (228, 'ä'),
    (229, 'å'),
    (230, 'æ'),
    (231, 'ç'),
    (232, 'è'),
    (233, 'é'),
    (234, 'ê'),
    (235, 'ë'),
    (236, 'ì'),
    (237, 'í'),
    (238, 'î'),
    (239, 'ï'),
    (240, 'ð'),
    (241, 'ñ'),
    (242, 'ò'),
    (243, 'ó'),
    (244, 'ô'),
    (245, 'õ'),
    (246, 'ö'),
    (247, '÷'),
    (248, 'ø'),
    (249, 'ù'),
    (250, 'ú'),
    (251, 'û'),
    (252, 'ü'),
    (253, 'ý'),
    (254, 'þ'),
    (255, 'ÿ'),
    (0, 'Ā'),
    (1, 'ā'),
    (2, 'Ă'),
    (3, 'ă'),
    (4, 'Ą'),
    (5, 'ą'),
    (6, 'Ć'),
    (7, 'ć'),
    (8, 'Ĉ'),
    (9, 'ĉ'),
    (10, 'Ċ'),
    (11, 'ċ'),
    (12, 'Č'),
    (13, 'č'),
    (14, 'Ď'),
    (15, 'ď'),
    (16, 'Đ'),
    (17, 'đ'),
    (18, 'Ē'),
    (19, 'ē'),
    (20, 'Ĕ'),
    (21, 'ĕ'),
    (22, 'Ė'),
    (23, 'ė'),
    (24, 'Ę'),
    (25, 'ę'),
    (26, 'Ě'),
    (27, 'ě'),
    (28, 'Ĝ'),
    (29, 'ĝ'),
    (30, 'Ğ'),
    (31, 'ğ'),
    (32, 'Ġ'),
    (127, 'ġ'),
    (128, 'Ģ'),
    (129, 'ģ'),
    (130, 'Ĥ'),
    (131, 'ĥ'),
    (132, 'Ħ'),
    (133, 'ħ'),
    (134, 'Ĩ'),
    (135, 'ĩ'),
    (136, 'Ī'),
    (137, 'ī'),
    (138, 'Ĭ'),
    (139, 'ĭ'),
    (140, 'Į'),
    (141, 'į'),
    (142, 'İ'),
    (143, 'ı'),
    (144, 'Ĳ'),
    (145, 'ĳ'),
    (146, 'Ĵ'),
    (147, 'ĵ'),
    (148, 'Ķ'),
    (149, 'ķ'),
    (150, 'ĸ'),
    (151, 'Ĺ'),
    (152, 'ĺ'),
    (153, 'Ļ'),
    (154, 'ļ'),
    (155, 'Ľ'),
    (156, 'ľ'),
    (157, 'Ŀ'),
    (158, 'ŀ'),
    (159, 'Ł'),
    (160, 'ł'),
    (173, 'Ń'),
];

const PAT: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    re: fancy_regex::Regex,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, String>,
    encoder: HashMap<String, u32>,
    decoder: Vec<String>,
    bpe_ranks: HashMap<(Vec<u8>, Vec<u8>), u32>,
}

impl Tokenizer {
    /// Creates a new tokenizer, this takes as input the path for the bpe rank file.
    pub fn new<T: AsRef<std::path::Path>>(vocab_path: T, merge_path: T) -> Result<Tokenizer> {
        let re = fancy_regex::Regex::new(PAT)?;
        let mut byte_encoder = [' '; 256];
        let mut byte_decoder = HashMap::new();
        for &(byte, unicode) in BYTES_TO_UNICODE.iter() {
            byte_decoder.insert(unicode, String::from_utf8_lossy(&[byte]).to_string());
            byte_encoder[byte as usize] = unicode
        }
        let encoder = std::fs::read_to_string(vocab_path)?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&encoder)?;
        let mut decoder = Vec::new();
        for (token_str, token_id) in encoder.iter() {
            let token_id = *token_id as usize;
            if token_id >= decoder.len() {
                decoder.resize(token_id + 1, "".to_string())
            }
            decoder[token_id] = token_str.clone()
        }
        let merge_file = std::fs::File::open(merge_path)?;
        let merge_file = std::io::BufReader::new(merge_file);
        let mut bpe_ranks = HashMap::new();
        for line in merge_file.lines() {
            let line = line?;
            let line = line.split(' ').collect::<Vec<_>>();
            if line.len() == 2 {
                let key = (line[0].as_bytes().to_vec(), line[1].as_bytes().to_vec());
                bpe_ranks.insert(key, bpe_ranks.len() as u32);
            }
        }
        Ok(Tokenizer { re, byte_decoder, byte_encoder, encoder, decoder, bpe_ranks })
    }

    /// The main tokenization entry point, takes as input a string and returns the list of tokens.
    pub fn encode(&self, s: &str) -> anyhow::Result<Vec<u32>> {
        let mut bpe_tokens: Vec<u32> = vec![];
        for word in self.re.find_iter(s) {
            let word = word?;
            let mut encoded_word = vec![];
            for &byte in word.as_str().as_bytes() {
                encoded_word.push(self.byte_encoder[byte as usize])
            }
            let encoded_word: String = encoded_word.iter().collect();
            bpe_tokens.extend(self.bpe(&encoded_word))
        }
        Ok(bpe_tokens)
    }

    fn get_pairs(word: &[Vec<u8>]) -> HashSet<(Vec<u8>, Vec<u8>)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, word: &str) -> Vec<u32> {
        let mut word: Vec<Vec<u8>> = word.chars().map(|x| x.to_string().into_bytes()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        while word.len() > 1 {
            let mut current_min = None;
            let pairs = Self::get_pairs(&word);
            for p in pairs.iter() {
                match self.bpe_ranks.get(p) {
                    None => {}
                    Some(v) => {
                        let should_replace = match current_min {
                            None => true,
                            Some((current_min, _)) => v < current_min,
                        };
                        if should_replace {
                            current_min = Some((v, p))
                        }
                    }
                }
            }
            let (first, second) = match current_min {
                None => break,
                Some((_v, (first, second))) => (first, second),
            };
            let mut new_word = vec![];
            let mut index = 0;
            while index < word.len() {
                let w = &word[index];
                if index + 1 < word.len() && w == first && &word[index + 1] == second {
                    let mut first_and_second = first.clone();
                    first_and_second.extend_from_slice(second);
                    new_word.push(first_and_second);
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter()
            .filter_map(|x| {
                let x = String::from_utf8_lossy(x).to_string();
                self.encoder.get(&x)
            })
            .copied()
            .collect()
    }

    /// The inverse of the tokenization process, takes as input a list of tokens and returns a
    /// string that produces this tokenization.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut str = vec![];
        for &token in tokens.iter() {
            let token = token as usize;
            if token >= self.decoder.len() {
                anyhow::bail!("token {token} is out of range {}", self.decoder.len())
            }
            str.push(self.decoder[token].as_str())
        }
        Ok(str.concat())
    }

    pub fn get_token(&self, s: &str) -> Option<u32> {
        self.encoder.get(s).copied()
    }

    // This should be memoized if it starts to be on the critical path.
    pub fn decode_token_id(&self, token_id: u32) -> Result<String> {
        let token_id = token_id as usize;
        if token_id >= self.decoder.len() {
            anyhow::bail!("token {token_id} is out of range {}", self.decoder.len())
        }
        let mut chars = vec![];
        for c in self.decoder[token_id].chars() {
            let c = match self.byte_decoder.get(&c) {
                None => c.to_string(),
                Some(s) => s.to_string(),
            };
            chars.push(c)
        }
        Ok(chars.concat())
    }
}
