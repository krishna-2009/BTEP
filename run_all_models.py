import os
import subprocess
import json
import numpy as np
import pandas as pd

# List of all models (configs and checkpoints from your directory)
models = [
    {"config": "cfa_r50_fpn_1x_dota_le135.py", "checkpoint": "cfa_r50_fpn_1x_dota_le135-aed1cbc6.pth"},
    {"config": "cfa_r50_fpn_40e_dota_oc.py", "checkpoint": "cfa_r50_fpn_40e_dota_oc-2f387232.pth"},
    {"config": "gliding_vertex_r50_fpn_1x_dota_le90.py", "checkpoint": "gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth"},
    {"config": "g_reppoints_r50_fpn_1x_dota_le135.py", "checkpoint": "g_reppoints_r50_fpn_1x_dota_le135-b840eed7.pth"},
    {"config": "oriented_rcnn_r50_fpn_1x_dota_le90.py", "checkpoint": "oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth"},
    {"config": "r3det_kfiou_ln_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d.pth"},
    {"config": "r3det_kld_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_kld_r50_fpn_1x_dota_oc-31866226.pth"},
    {"config": "r3det_kld_stable_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_kld_stable_r50_fpn_1x_dota_oc-e011059d.pth"},
    {"config": "r3det_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_r50_fpn_1x_dota_oc-b1fb045c.pth"},
    {"config": "r3det_tiny_kld_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_tiny_kld_r50_fpn_1x_dota_oc-589e142a.pth"},
    {"config": "r3det_tiny_kld_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_tiny_kld_r50_fpn_1x_dota_oc-589e142a.pth"},
    {"config": "r3det_tiny_r50_fpn_1x_dota_oc.py", "checkpoint": "r3det_tiny_r50_fpn_1x_dota_oc-c98a616c.pth"},
    {"config": "redet_re50_refpn_1x_dota_le90.py", "checkpoint": "redet_re50_fpn_1x_dota_le90-724ab2da.pth"},
    {"config": "redet_re50_refpn_1x_dota_ms_rr_le90.py", "checkpoint": "redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth"},
    {"config": "redet_re50_refpn_fp16_1x_dota_le90.py", "checkpoint": "redet_re50_refpn_fp16_1x_dota_le90-1e34da2d.pth"},
    {"config": "roi_trans_r50_fpn_1x_dota_le90.py", "checkpoint": "roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth"},
    {"config": "roi_trans_r50_fpn_fp16_1x_dota_le90.py", "checkpoint": "roi_trans_r50_fpn_fp16_1x_dota_le90-62eb88b1.pth"},
    {"config": "roi_trans_swin_tiny_fpn_1x_dota_le90.py", "checkpoint": "roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth"},
    {"config": "rotated_atss_hbb_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_atss_hbb_r50_fpn_1x_dota_oc-eaa94033.pth"},
    {"config": "rotated_atss_obb_r50_fpn_1x_dota_le135.py", "checkpoint": "rotated_atss_obb_r50_fpn_1x_dota_le135-eab7bc12.pth"},
    {"config": "rotated_atss_obb_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_atss_obb_r50_fpn_1x_dota_le90-e029ca06.pth"},
    {"config": "rotated_faster_rcnn_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth"},
    {"config": "rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90-4e044ad2.pth"},
    {"config": "rotated_fcos_kld_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_fcos_kld_r50_fpn_1x_dota_le90-ecafdb2b.pth"},
    {"config": "rotated_fcos_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth"},
    {"config": "rotated_fcos_sep_angle_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth"},
    {"config": "rotated_reppoints_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth"},
    {"config": "rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc-41fd7805.pth"},
    {"config": "rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135.py", "checkpoint": "rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135-0eaa4156.pth"},
    {"config": "rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90-03e02f75.pth"},
    {"config": "rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc-c00be030.pth"},
    {"config": "rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc-49c1f937.pth"},
    {"config": "rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc-92a76443.pth"},
    {"config": "rotated_retinanet_hbb_kld_stable_r50_fpn_6x_hrsc_rr_oc.py", "checkpoint": "rotated_retinanet_hbb_kld_stable_r50_fpn_6x_hrsc_rr_oc-9a4ac8e2.pth"},
    {"config": "rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py", "checkpoint": "rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth"},
    {"config": "rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc.py", "checkpoint": "rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc-f37eada6.pth"},
    {"config": "rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90.py", "checkpoint": "rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90-b4271aed.pth"},
    {"config": "rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90.py", "checkpoint": "rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90-474d9955.pth"},
    {"config": "rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90-31193e00.pth"},
    {"config": "rotated_retinanet_obb_kld_stable_r50_fpn_6x_hrsc_rr_le90.py", "checkpoint": "rotated_retinanet_obb_kld_stable_r50_fpn_6x_hrsc_rr_le90-58665364.pth"},
    {"config": "rotated_retinanet_obb_r50_fpn_1x_dota_le135.py", "checkpoint": "rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth"},
    {"config": "rotated_retinanet_obb_r50_fpn_1x_dota_le90.py", "checkpoint": "rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth"},
    {"config": "rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py", "checkpoint": "rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py"},
    {"config": "rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py", "checkpoint": "rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90-ee4f18af.pth"},
    {"config": "rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py", "checkpoint": "rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth"},
    {"config": "s2anet_r50_fpn_1x_dota_le135.py", "checkpoint": "s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth"},
    {"config": "s2anet_r50_fpn_fp16_1x_dota_le135.py", "checkpoint": "s2anet_r50_fpn_fp16_1x_dota_le135-5cac515c.pth"}
]

class ModelEvaluator:
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.results = {}
        os.makedirs(result_dir, exist_ok=True)

    def evaluate_model(self, config_file, checkpoint_file, image_path):
        # Prepare the output file path
        output_file = os.path.join(self.result_dir, f"{config_file.split('.')[0]}_result.jpg")
        
        # Run inference using the demo command and capture the output
        command = f"python demo/image_demo.py {image_path} {config_file} {checkpoint_file} --out-file {output_file}"
        try:
            result = subprocess.check_output(command, shell=True)
            # Parse the result
            result = json.loads(result.decode('utf-8'))
        except subprocess.CalledProcessError as e:
            print(f"Error running command for {config_file}: {str(e)}")
            return None
        except json.JSONDecodeError:
            print(f"Error parsing JSON output for {config_file}")
            return None
        
        # Get model name from config
        model_name = os.path.splitext(os.path.basename(config_file))[0]
        
        # Process results
        total_detections = 0
        confidence_scores = []
        
        # Handle different result formats
        for class_results in result:
            if len(class_results) > 0:
                total_detections += len(class_results)
                confidence_scores.extend([res[-1] for res in class_results])
        
        # Store results
        self.results[model_name] = {
            'total_detections': total_detections,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0
        }
        
        return self.results[model_name]

    def generate_comparison_report(self):
        # Convert results to DataFrame for easy comparison
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Sort by total detections and confidence
        df_by_detections = df.sort_values('total_detections', ascending=False)
        df_by_confidence = df.sort_values('avg_confidence', ascending=False)
        
        # Save reports
        df.to_csv(os.path.join(self.result_dir, 'model_comparison.csv'))
        
        # Generate summary report
        with open(os.path.join(self.result_dir, 'summary_report.txt'), 'w') as f:
            f.write("Model Comparison Summary\n")
            f.write("======================\n\n")
            
            f.write("Top 5 Models by Detection Count:\n")
            f.write(df_by_detections.head().to_string())
            f.write("\n\n")
            
            f.write("Top 5 Models by Average Confidence:\n")
            f.write(df_by_confidence.head().to_string())
            
        return df

def main():
    image_path = 'demo/1.jpg'
    result_dir = 'demo/results'
    
    evaluator = ModelEvaluator(result_dir)
    
    # Run evaluation for each model
    for model in models:
        print(f"Processing {model['config']}...")
        try:
            results = evaluator.evaluate_model(model['config'], 
                                            model['checkpoint'], 
                                            image_path)
            if results:
                print(f"Detected {results['total_detections']} objects with average confidence {results['avg_confidence']:.3f}")
            else:
                print(f"No results for {model['config']}")
        except Exception as e:
            print(f"Error processing {model['config']}: {str(e)}")
    
    # Generate comparison report
    comparison_df = evaluator.generate_comparison_report()
    print("\nEvaluation complete. Check the results directory for detailed reports.")

if __name__ == '__main__':
    main()
