// https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-CPP-Inference/inference.h
// https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-CPP-Inference/inference.cpp

#![allow(dead_code)]

use rand::prelude::*;
use std::ffi::c_void;

use opencv::core as cv;
use opencv::dnn::{self, Net};
use opencv::prelude::*;
use opencv::Result;

pub const CLASSES: [&str; 80] = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

#[derive(Debug)]
pub struct Detection {
  pub class_id: i32,
  pub class_name: String,
  pub confidence: f32,
  pub color: cv::Scalar,
  pub box_: cv::Rect, // Rust does not allow box as field name
}

impl Default for Detection {
  fn default() -> Self {
    Self {
      class_id: 0,
      class_name: "".to_string(),
      confidence: 0.0,
      color: cv::Scalar::default(),
      box_: cv::Rect::default(),
    }
  }
}

pub struct Interface {
  model_path: String,
  classes_path: String,

  model_shape: cv::Size,

  model_confidence_threshold: f32,
  model_score_threshold: f32,
  model_nms_threshold: f32,

  letter_box_for_square: bool,

  net: Net,
}

impl Default for Interface {
  fn default() -> Self {
    Self {
      model_path: "".to_string(),
      classes_path: "".to_string(),
      model_shape: cv::Size::default(),
      model_confidence_threshold: 0.0,
      model_score_threshold: 0.0,
      model_nms_threshold: 0.0,
      letter_box_for_square: true,
      net: Net::default().unwrap(),
    }
  }
}

impl Interface {
  pub fn new(
    onnx_model_path: &str,
    model_input_shape: cv::Size,
    classes_txt_file: &str,
    run_with_cuda: bool,
  ) -> Result<Self> {
    let mut net = dnn::read_net_from_onnx(onnx_model_path)?;
    // let mut net = dnn::read_net_from_onnx(onnx_model_path)?;

    if run_with_cuda {
      println!("Running on CUDA");
      net.set_preferable_backend(dnn::DNN_BACKEND_CUDA)?;
      net.set_preferable_target(dnn::DNN_TARGET_CUDA)?;
    } else {
      println!("Running on CPU");
      net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
      net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
    }

    Ok(Self {
      model_path: onnx_model_path.to_string(),
      model_shape: model_input_shape,
      classes_path: classes_txt_file.to_string(),
      net,
      ..Default::default()
    })
  }

  pub fn run_interface(&mut self, input: &Mat) -> Result<Vec<Detection>> {
    let mut rng = rand::thread_rng();
    let mut model_input = input.clone();

    if self.letter_box_for_square && self.model_shape.width == self.model_shape.height {
      model_input = Interface::format_to_square(&model_input)?;
    }

    let blob = dnn::blob_from_image(
      &model_input,
      1.0 / 255.0,
      self.model_shape,
      cv::Scalar::default(),
      true,
      false,
      cv::CV_32F,
    )?;
    self.net.set_input_def(&blob)?;

    let mut outputs: cv::Vector<Mat> = cv::Vector::new();
    self
      .net
      .forward(&mut outputs, &self.net.get_unconnected_out_layers_names()?)?;
    let mut output = outputs.get(0)?;

    let mut rows = output.mat_size().get(1)?;
    let mut dimensions = output.mat_size().get(2)?;

    if dimensions > rows {
      rows = output.mat_size().get(2)?;
      dimensions = output.mat_size().get(1)?;

      output = output.reshape(1, dimensions)?;
      cv::transpose(&output.clone(), &mut output)?;
    }

    let mut data = output.data();
    // println!("Data: {data:?}, Value: {}", unsafe { *data });
    // println!("Typed Data: {typed_data:?}");

    let x_factor = (model_input.cols() / self.model_shape.width) as f32;
    let y_factor = (model_input.rows() / self.model_shape.height) as f32;

    let mut class_ids: cv::Vector<i32> = cv::Vector::new();
    let mut confidences: cv::Vector<f32> = cv::Vector::new();
    let mut boxes: cv::Vector<cv::Rect> = cv::Vector::new();

    for _ in 0..rows {
      let typed_data: &[f32] = output.data_typed()?;
      let classes_scores = unsafe { data.add(4) };
      let scores = unsafe {
        Mat::new_rows_cols_with_data_def(
          1,
          CLASSES.len() as i32,
          cv::CV_32FC1,
          classes_scores as *mut c_void,
        )
      }?;
      let mut class_id = cv::Point::default();
      let mut max_class_score = f64::default();

      cv::min_max_loc(
        &scores,
        None,
        Some(&mut max_class_score),
        None,
        Some(&mut class_id),
        &cv::no_array(),
      )?;

      if max_class_score as f32 > self.model_score_threshold {
        confidences.push(max_class_score as f32);
        class_ids.push(class_id.x);

        // println!("{:?}", typed_data.get(0..10));
        // FIXME: Coordinates comes wrong
        let (x, y, w, h) = (typed_data[0], typed_data[1], typed_data[2], typed_data[3]);

        // println!("X: {x}, Y: {y}, W: {w}, H: {h}");

        let left = ((x - w / 2f32) * x_factor) as i32;
        let top = ((y - h / 2f32) * y_factor) as i32;

        let width = (w * x_factor) as i32;
        let height = (h * y_factor) as i32;

        boxes.push(cv::Rect::new(left, top, width, height));
      }

      data = unsafe { data.add(dimensions as usize) };
      unsafe { output.set_data(data) };
    }

    let mut nms_result: cv::Vector<i32> = cv::Vector::new();
    dnn::nms_boxes_def(
      &boxes,
      &confidences,
      self.model_score_threshold,
      self.model_nms_threshold,
      &mut nms_result,
    )?;

    let mut detections: Vec<Detection> = Vec::new();

    for idx in nms_result {
      let class_id = class_ids.get(idx as usize)?;
      let class_name = CLASSES[class_id as usize].to_string();
      let confidence = confidences.get(idx as usize)?;
      let color = cv::Scalar::new(
        rng.gen_range(100..=255) as f64,
        rng.gen_range(100..=255) as f64,
        rng.gen_range(100..=255) as f64,
        0.0,
      );
      let box_ = boxes.get(idx as usize)?;

      detections.push(Detection {
        class_id,
        class_name,
        confidence,
        color,
        box_,
      })
    }

    Ok(detections)
  }

  fn format_to_square(source: &Mat) -> Result<Mat> {
    let col = source.cols();
    let row = source.rows();
    let max = std::cmp::max(row, col);

    let mut result = Mat::zeros(max, max, cv::CV_8UC3)?
      .apply_1(cv::Rect::new(0, 0, col, row))?
      .to_mat()?;
    source.copy_to(&mut result)?;

    Ok(result)
  }
}
