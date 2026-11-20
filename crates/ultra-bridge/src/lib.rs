//! Temporary research bridge. No deployment or profile qualification is implied.

pub mod engine;
pub mod nemotron;
pub mod nemotron_service;
pub mod protocol;
pub mod records;
pub mod run;
pub mod service;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(std::io::Error::other(message.into()).into())
    }
}
