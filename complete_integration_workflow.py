# Enhanced Integration Example: Complete Workflow
# Combining Standardized Framework with Co-occurrence Based Overlap Control

import os
import json
import logging
from typing import List, Dict
import argparse
import pandas as pd
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_SPLIT_FILE = "splits_json_latest" #splits_json
CUSTOM_STEP_SPLITS_DIR= "custom_overlap_splits_latest"


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def complete_integration_workflow(root_dir: str = "datasets",
                                output_dir: str = JSON_SPLIT_FILE, 
                                scenarios: List[str] = None,
                                img_overlap_ratios: List[int] = None,
                                cls_overlap_ratios: List[int] = None,
                                apply_cooccurrence_control: bool = True,
                                randomize_class_order: bool = False,
                                custom_step_splits_dir: str = "custom_overlap_splits_latest",
                                class_order: List[int] = list(range(0,150))):
    """
    Complete workflow integrating standardized dataset preparation 
    with co-occurrence based overlap control.
    
    This demonstrates the full integration of:
    1. Standardized dataset preparation framework
    2. Co-occurrence matrix based overlap measurement  
    3. Intelligent image reorganization for precise overlap control
    4. Multiple overlap ratio generation
    5. Evaluation protocol compatibility
    
    Args:
        root_dir: Root directory containing ADE20K dataset
        output_dir: Output directory for enhanced splits
        scenarios: List of scenarios to generate (e.g., ["100-10", "100-5"])
        apply_cooccurrence_control: Whether to apply co-occurrence based control
    """
    
    if scenarios is None:
        scenarios = ["100-10", "100-5", "50-50"]
        
    if img_overlap_ratios is None:                                           # *** NEW ***
        img_overlap_ratios = [100, 100, 75, 75, 50 ,50, 25, 25, 0]
        
    if cls_overlap_ratios is None:                                           # *** NEW ***
        cls_overlap_ratios = [76,  75 , 70, 66, 68 ,55, 73, 69 ,75]
    
    logger.info("=== Enhanced Continual Learning Dataset Preparation ===")
    logger.info("Integrating standardized framework with co-occurrence overlap control")
    
    # Import our custom modules (would be actual imports in practice)
    try:
        from shared.standard_prepare_datasets import StandardizedDatasetPreparator
        from tools.cooccurrence_overlap_control import CooccurrenceAnalyzer, OverlapConfig, integrate_with_standardized_framework
        from shared.optimized_continual_dataloader import ContinualSegmentationDataLoader
        from shared.standardized_evaluator import StandardizedContinualEvaluator
    except ImportError as e:
        logger.error(e)
        logger.error("Please ensure all framework modules are in Python path")
        logger.info("Required modules: standardized_prepare_datasets.py, cooccurrence_overlap_control.py, etc.")
        return
    
    # Step 1: Configure co-occurrence based overlap control
    logger.info("Step 1: Configuring co-occurrence based overlap control")
    
    overlap_configs = {
        'precise_control': OverlapConfig(
            threshold_k=100,                 # Images with >100 co-occurrences considered "common"
            threshold_p=100,                 # Use same threshold for re-splitting
            threshold_h=50,                  # Minimum image count threshold to avoid rare categories
            target_reduction_factor=1.0,     # Will be set per target ratio in workflow
            max_iterations=500,
            min_swap_images=20,
            swap_priority=('test', 'base'),  # Always swap to test set first, then base set
            enable_dataset_balancing=True,   # Enable fair B/C balancing
            random_seed=42
        ),
        'aggressive_control': OverlapConfig(
            threshold_k=50,                  # Images with >50 co-occurrences considered "common"
            threshold_p=50,                  # Lower re-splitting threshold for aggressiveness
            threshold_h=20,                  # Allow rarer categories when balancing
            target_reduction_factor=1.0,
            max_iterations=500,
            min_swap_images=20,
            swap_priority=('test', 'base'),
            enable_dataset_balancing=True,
            random_seed=42
        ),
        
        'super_aggressive_control': OverlapConfig(
            threshold_k=25,                  # Images with >25 co-occurrences considered "common"
            threshold_p=25,                  # Lower re-splitting threshold for aggressiveness
            threshold_h=10,                  # Allow rarer categories when balancing
            target_reduction_factor=1.0,
            max_iterations=500,
            min_swap_images=20,
            swap_priority=('test', 'base'),
            enable_dataset_balancing=True,
            random_seed=42
        ),
        
        'ultra_aggressive_control': OverlapConfig(
            threshold_k=5,                  # Images with >5 co-occurrences considered "common"
            threshold_p=5,                  # Lower re-splitting threshold for aggressiveness
            threshold_h=2,                  # Allow rarer categories when balancing
            target_reduction_factor=1.0,
            max_iterations=500,
            min_swap_images=20,
            swap_priority=('test', 'base'),
            enable_dataset_balancing=True,
            random_seed=42
        )
    }
    
    # Step 2: Initialize standardized preparator
    logger.info("Step 2: Initializing standardized dataset preparator")
    
    preparator = StandardizedDatasetPreparator(
        root_dir=root_dir,
        output_dir=output_dir,
        img_overlap_ratio = 100,
        cls_overlap_ratio = 100, # Start with original overlap
        random_seed=42,
        class_order =class_order,
        randomize_class_order = randomize_class_order
    )
    
    # Step 3: Load and analyze original dataset
    logger.info("Step 3: Loading and analyzing original ADE20K dataset")
    
    train_data, val_data = preparator.load_dataset_info()
    val_split, test_split = preparator.create_standardized_test_split(val_data)
    
    logger.info(f"Dataset loaded: {len(train_data['images'])} train, {len(val_split['images'])} val, {len(test_split['images'])} test")
    
    # Step 4: Generate enhanced splits for each scenario
    logger.info("Step 4: Generating enhanced splits with co-occurrence control")
    
    
    
    
    for scenario in scenarios:
        logger.info(f"\\n--- Processing scenario: {scenario} ---")
        
 
        
        if apply_cooccurrence_control:
            # Option A: Enhanced generation with co-occurrence control
            enhanced_preparator = integrate_with_standardized_framework(
                preparator, 
                overlap_configs['ultra_aggressive_control'],
                target_overlap_ratios=img_overlap_ratios  
            )
            
            # Generate with mathematical overlap control
            enhanced_preparator.enhanced_generate_scenario_splits(
                scenario=scenario,
                train_data=train_data,
                val_split=val_split,
                test_split=test_split,
                apply_overlap_control=True
            )
            
            logger.info(f"Generated {scenario} with co-occurrence based overlap control")
            
        else:
            # Option B: Standard generation with probabilistic overlap
            
            for img_ov, cls_ov in  zip(img_overlap_ratios, cls_overlap_ratios):
                
                
                # Fall back to original method
                preparator.img_overlap_ratio = img_ov 
                preparator.cls_overlap_ratio = cls_ov
                preparator.custom_step_splits_dir =custom_step_splits_dir
                
                
                preparator.generate_scenario_splits(
                    scenario, train_data, val_split, test_split
                )
            
            logger.info(f"Generated {scenario} with standard probabilistic overlap")
    


    
    logger.info("\\n=== Integration Workflow Completed Successfully ===")
    logger.info(f"Enhanced datasets available in: {output_dir}")
    logger.info("Ready for multi-method evaluation with precise overlap control!")

def analyze_generated_splits(output_dir: str, scenarios: List[str], with_cooccurrence: bool) -> Dict:
    """Analyze the generated splits and provide statistics."""
    
    analysis = {
        'scenarios': scenarios,
        'overlap_methods': [],
        'split_statistics': {},
        'overlap_analysis': {}
    }
    
    if with_cooccurrence:
        analysis['overlap_methods'] = ['cooccurrence_100', 'cooccurrence_75', 'cooccurrence_50', 'cooccurrence_25', 'cooccurrence_0']
    else:
        analysis['overlap_methods'] = ['overlap_100', 'overlap_75', 'overlap_50', 'overlap_25', 'overlap_0']
    
    # Analyze each scenario
    for scenario in scenarios:
        analysis['split_statistics'][scenario] = {}
        
        # Count files for each overlap method
        panoptic_dir = os.path.join(output_dir, 'panoptic')
        if os.path.exists(panoptic_dir):
            for overlap_method in analysis['overlap_methods']:
                pattern_files = [f for f in os.listdir(panoptic_dir) 
                               if scenario in f and overlap_method in f]
                analysis['split_statistics'][scenario][overlap_method] = len(pattern_files)
    
    # Load overlap metadata if available
    metadata_dir = os.path.join(output_dir, 'splits_metadata')
    if os.path.exists(metadata_dir):
        for scenario in scenarios:
            metadata_files = [f for f in os.listdir(metadata_dir) 
                            if scenario in f and 'overlap_metadata.json' in f]
            
            analysis['overlap_analysis'][scenario] = {}
            for metadata_file in metadata_files:
                try:
                    with open(os.path.join(metadata_dir, metadata_file), 'r') as f:
                        metadata = json.load(f)
                        overlap_type = metadata_file.split('_')[2]  # Extract overlap type
                        analysis['overlap_analysis'][scenario][overlap_type] = {
                            'original_eta': metadata.get('original_eta', 0),
                            'final_eta': metadata.get('final_eta', 0),
                            'swaps_performed': metadata.get('swaps_performed', 0)
                        }
                except Exception as e:
                    logger.warning(f"Could not load metadata from {metadata_file}: {e}")
    
    return analysis

def setup_enhanced_dataloaders(output_dir: str, scenarios: List[str]) -> Dict:
    """Setup optimized dataloaders for the enhanced splits."""
    
    dataloader_configs = {}
    
    for scenario in scenarios:
        dataloader_configs[scenario] = {
            'standard_overlaps': {},
            'cooccurrence_overlaps': {}
        }
        
        # Setup for different overlap methods
        for overlap_ratio in [100, 75, 50, 25, 0]:
            # Standard probabilistic overlap
            dataloader_configs[scenario]['standard_overlaps'][f'overlap_{overlap_ratio}'] = {
                'data_root': output_dir,
                'overlap_ratio': overlap_ratio,
                'batch_size': 'auto',
                'num_workers': 'auto'
            }
            
            # Co-occurrence based overlap
            dataloader_configs[scenario]['cooccurrence_overlaps'][f'cooccurrence_{overlap_ratio}'] = {
                'data_root': output_dir,
                'overlap_ratio': overlap_ratio,
                'method': 'cooccurrence_controlled',
                'batch_size': 'auto', 
                'num_workers': 'auto'
            }
    
    # Save dataloader configuration
    config_path = os.path.join(output_dir, 'enhanced_dataloader_config.json')
    with open(config_path, 'w') as f:
        json.dump(dataloader_configs, f, indent=2)
    
    logger.info(f"Dataloader configurations saved to: {config_path}")
    
    return dataloader_configs

def setup_evaluation_framework(output_dir: str, scenarios: List[str], with_cooccurrence: bool) -> Dict:
    """Setup evaluation framework for enhanced splits."""
    
    evaluation_setup = {
        'scenarios': scenarios,
        'evaluation_protocols': [],
        'comparison_matrices': {},
        'expected_outputs': {}
    }
    
    # Define evaluation protocols
    if with_cooccurrence:
        evaluation_setup['evaluation_protocols'] = [
            'standard_overlap_comparison',
            'cooccurrence_overlap_comparison', 
            'cross_method_comparison',
            'overlap_sensitivity_analysis'
        ]
    else:
        evaluation_setup['evaluation_protocols'] = [
            'standard_overlap_comparison',
            'probabilistic_overlap_analysis'
        ]
    
    # Setup comparison matrices for each scenario
    for scenario in scenarios:
        evaluation_setup['comparison_matrices'][scenario] = {
            'methods': ['method1', 'method2', 'method3', 'method4'],
            'overlap_settings': ['100%', '75%', '50%', '25%', '0%'],
            'metrics': ['final_PQ', 'final_PQ_th', 'final_PQ_st', 'average_forgetting'],
            'control_methods': ['probabilistic', 'cooccurrence'] if with_cooccurrence else ['probabilistic']
        }
    
    # Expected outputs
    evaluation_setup['expected_outputs'] = {
        'comparison_tables': [f"{scenario}_overlap_comparison.csv" for scenario in scenarios],
        'analysis_plots': [f"{scenario}_overlap_sensitivity.png" for scenario in scenarios],
        'latex_tables': [f"{scenario}_results_table.tex" for scenario in scenarios],
        'comprehensive_report': 'enhanced_evaluation_report.md'
    }
    
    return evaluation_setup


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Enhanced Continual Learning Dataset Integration")
    parser.add_argument("--root_dir", default="datasets", help="Root directory containing ADE20K")
    parser.add_argument("--output_dir", default=JSON_SPLIT_FILE, help="Output directory")
    parser.add_argument("--scenarios", nargs='+', default=["100-5", "100-10", "100-50", "50-10","50-20", "50-50"], 
                       help="Scenarios to generate")
    parser.add_argument("--img_overlap_ratios", nargs='+', default=[100, 75, 75,  50, 50, 25, 25, 0],
                       help="Overlap ratios to generate (e.g., --img_overlap_ratio 1.0 0.5 0.0)")
    parser.add_argument("--cls_overlap_ratios", nargs='+', default=[75,  73, 68, 71, 64, 73, 69, 75],
                       help="Overlap ratios to generate (e.g., --cls_overlap_ratio 1.0 0.5 0.0)")
    parser.add_argument("--enable_cooccurrence", action='store_true', 
                       help="Enable co-occurrence based overlap control")
    parser.add_argument("--disable_cooccurrence", action='store_true',
                       help="Disable co-occurrence control (use probabilistic only)")
    parser.add_argument("--class_order", nargs='+', type=List[int], default=list(range(0,150)),
                       help="Order of classes (e.g., --class_order 0, 1, 2, ..)")
    parser.add_argument("--randomize_class_order", action='store_true',  default=False,
                       help="Whether to randomize the classes")
    parser.add_argument("--custom_step_splits_dir", default=CUSTOM_STEP_SPLITS_DIR, help="custom step split dir")
    
    
    args = parser.parse_args()
    
    # Determine co-occurrence setting
    if args.disable_cooccurrence:
        apply_cooccurrence = False
    elif args.enable_cooccurrence:
        apply_cooccurrence = True  
    else:
        # Default: enable co-occurrence control
        apply_cooccurrence = False
    

    logger.info(
        f"STARTING COMPLETE INTEGRATION WORKFLOW WITH:\n"
        f"SCENARIOS: {args.scenarios}\n"
        f"Image OVERLAP RATIOS: {args.img_overlap_ratios}\n" 
        f"Class OVERLAP RATIOS: {args.cls_overlap_ratios}\n" 
        f"COOCCURRENCE CONTROL: {apply_cooccurrence}\n"
        f"RANDOMIZE CLASS ORDER: {args.randomize_class_order}\n"
        f"CLASS ORDER: {args.class_order}"
    )
    
    
    
    # Run complete integration workflow
    complete_integration_workflow(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        scenarios=args.scenarios,                    # *** PASS SCENARIOS ARRAY ***
        img_overlap_ratios=args.img_overlap_ratios,
        cls_overlap_ratios=args.cls_overlap_ratios,# *** PASS OVERLAP RATIOS ARRAY ***
        apply_cooccurrence_control=apply_cooccurrence,
        class_order =args.class_order,
        custom_step_splits_dir = args.custom_step_splits_dir,
        randomize_class_order = args.randomize_class_order
    )

if __name__ == "__main__":
    main()