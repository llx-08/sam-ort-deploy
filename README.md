# SAM-ort-deploy
Export and depoly SAM(segment-anything model) on onnxruntime.

## Export onnx model
SAM repo: https://github.com/facebookresearch/segment-anything
1. Install SAM via readme in SAM.
2. Download SAM checkpoint & Set the path of checkpoint.
3. Run python script in dir export_onnx, the sam is composed of two parts: **image encoder** and **mask decoder**, we will export them into onnx model separately

To export mask decoder, run 
   



## Run Inference
Run ort_inference.py