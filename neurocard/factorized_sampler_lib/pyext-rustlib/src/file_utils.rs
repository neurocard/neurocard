use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

const CACHE_DIR: &str = "./cache/";

fn get_path(filename: &str) -> String {
    let buf: PathBuf = [CACHE_DIR, filename].iter().collect();
    buf.as_path().to_str().unwrap().to_string()
}

pub fn save<T: serde::Serialize>(filename: &str, data: &T) {
    let path = get_path(filename);
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    bincode::serialize_into(&mut writer, data).unwrap();
}

pub fn load<T: serde::de::DeserializeOwned>(filename: &str) -> T {
    let path = get_path(filename);
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    bincode::deserialize_from(reader).unwrap()
}
