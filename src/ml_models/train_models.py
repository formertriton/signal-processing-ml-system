"""
Model Training Script
Trains all ML models and generates evaluation reports
"""

from model_trainer import SignalClassifier


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("SIGNAL CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Initialize classifier
    classifier = SignalClassifier(random_state=42)
    
    # Load data
    classifier.load_data()
    
    # Prepare data (split and scale)
    classifier.prepare_data(test_size=0.2, val_size=0.1)
    
    # Train all models
    classifier.train_all_models()
    
    # Evaluate each model
    print("\n" + "="*60)
    print("GENERATING EVALUATION REPORTS")
    print("="*60)
    
    for model_name in classifier.models.keys():
        classifier.evaluate_model(model_name, save_plots=True)
    
    # Compare all models
    classifier.compare_models(save_plot=True)
    
    # Save trained models
    classifier.save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - Confusion matrices: results/")
    print("  - Model comparison: results/model_comparison.png")
    print("  - Trained models: models/trained/")
    
    # Print best model
    best_model = max(classifier.results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\nBest performing model: {best_model[0].upper()}")
    print(f"Test accuracy: {best_model[1]['test_acc']*100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run extract_features.py first!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()