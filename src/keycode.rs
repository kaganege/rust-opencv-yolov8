#![allow(dead_code)]

use std::fmt;
pub struct KeyCode(pub i32);

impl KeyCode {
  pub fn is_valid(&self) -> bool {
    if self.0 < u32::MIN as i32 {
      false
    } else {
      true
    }
  }

  pub fn to_char(&self) -> Option<char> {
    if self.is_valid() {
      char::from_u32(u32::try_from(self.0).unwrap())
    } else {
      None
    }
  }
}

impl PartialEq for KeyCode {
  fn eq(&self, other: &Self) -> bool {
    self.0 == other.0
  }
}

impl fmt::Display for KeyCode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

// TODO: Add an enum of keys
