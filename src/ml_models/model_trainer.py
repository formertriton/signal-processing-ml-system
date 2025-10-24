"""
Machine Learning Model Training Module
Trains multiple classifiers for signal classification
"""

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib


class SignalClassifier:
    """
    Trains and evaluates multiple ML models for signal classification.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_data(self, data_path='data/processed/processed_features_latest.pkl'):
        """
        Load processed features.
        
        Args:
            data_path: Path to processed feature file
        """
        print(f"Loading processed data from: {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.features = data['features']
        self.labels = data['labels']
        self.feature_names = data['feature_names']
        
        print(f"✓ Data loaded: {self.features.shape[0]} samples, {self.features.shape[1]} features")
        print(f"✓ Signal types: {np.unique(self.labels)}")
        
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """
        Split and scale the data.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
        """
        print("\nPreparing data...")
        
        # Encode labels (convert text to numbers)
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.label_names = self.label_encoder.classes_
        
        print(f"  Label encoding: {dict(zip(self.label_names, range(len(self.label_names))))}")
        
        # Split into train and temp (test + val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.labels_encoded,
            test_size=(test_size + val_size),
            random_state=self.random_state,
            stratify=self.labels_encoded
        )
        
        # Split temp into test and validation
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Scale features (normalize to same range)
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"✓ Data split complete:")
        print(f"    Training: {len(self.X_train)} samples")
        print(f"    Validation: {len(self.X_val)} samples")
        print(f"    Test: {len(self.X_test)} samples")
        
    def train_random_forest(self, n_estimators=100, max_depth=20):
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Training...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"✓ Training complete!")
        print(f"    Train accuracy: {train_acc*100:.2f}%")
        print(f"    Validation accuracy: {val_acc*100:.2f}%")
        print(f"    Test accuracy: {test_acc*100:.2f}%")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }
        
        return model
    
    def train_svm(self, kernel='rbf', C=1.0):
        """
        Train Support Vector Machine.
        
        Args:
            kernel: Kernel type
            C: Regularization parameter
        """
        print("\n" + "="*60)
        print("TRAINING SUPPORT VECTOR MACHINE")
        print("="*60)
        
        model = SVC(
            kernel=kernel,
            C=C,
            random_state=self.random_state
        )
        
        print("Training (this may take a few minutes)...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"✓ Training complete!")
        print(f"    Train accuracy: {train_acc*100:.2f}%")
        print(f"    Validation accuracy: {val_acc*100:.2f}%")
        print(f"    Test accuracy: {test_acc*100:.2f}%")
        
        self.models['svm'] = model
        self.results['svm'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }
        
        return model
    
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1):
        """
        Train Gradient Boosting classifier.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
        """
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING")
        print("="*60)
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        
        print("Training...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"✓ Training complete!")
        print(f"    Train accuracy: {train_acc*100:.2f}%")
        print(f"    Validation accuracy: {val_acc*100:.2f}%")
        print(f"    Test accuracy: {test_acc*100:.2f}%")
        
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }
        
        return model
    
    def train_neural_network(self, hidden_layers=(100, 50), max_iter=200):
        """
        Train Neural Network (MLP).
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            max_iter: Maximum iterations
        """
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK")
        print("="*60)
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        print("Training...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"✓ Training complete!")
        print(f"    Train accuracy: {train_acc*100:.2f}%")
        print(f"    Validation accuracy: {val_acc*100:.2f}%")
        print(f"    Test accuracy: {test_acc*100:.2f}%")
        
        self.models['neural_network'] = model
        self.results['neural_network'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }
        
        return model
    
    def train_all_models(self):
        """Train all available models."""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_neural_network()
        self.train_svm()
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED!")
        print("="*60)
        
    def evaluate_model(self, model_name, save_plots=True, output_dir='results'):
        """
        Detailed evaluation of a specific model.
        
        Args:
            model_name: Name of model to evaluate
            save_plots: Whether to save confusion matrix plot
            output_dir: Directory to save plots
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.label_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_names,
                       yticklabels=self.label_names)
            plt.title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Confusion matrix saved to: {save_path}")
    
    def compare_models(self, save_plot=True, output_dir='results'):
        """
        Compare performance of all models.
        
        Args:
            save_plot: Whether to save comparison plot
            output_dir: Directory to save plot
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train': results['train_acc'] * 100,
                'Validation': results['val_acc'] * 100,
                'Test': results['test_acc'] * 100
            })
        
        # Print table
        print(f"\n{'Model':<25} {'Train':<10} {'Val':<10} {'Test':<10}")
        print("-" * 55)
        for data in comparison_data:
            print(f"{data['Model']:<25} {data['Train']:<10.2f} {data['Validation']:<10.2f} {data['Test']:<10.2f}")
        
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create comparison bar plot
            models = [d['Model'] for d in comparison_data]
            test_accs = [d['Test'] for d in comparison_data]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            plt.ylabel('Test Accuracy (%)', fontsize=12)
            plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
            plt.ylim([0, 100])
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'model_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ Comparison plot saved to: {save_path}")
    
    def save_models(self, output_dir='models/trained'):
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nSaving models to: {output_dir}")
        
        for model_name, model in self.models.items():
            filename = f"{model_name}_{timestamp}.pkl"
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"  ✓ {model_name} saved")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, os.path.join(output_dir, f'scaler_{timestamp}.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, f'label_encoder_{timestamp}.pkl'))
        
        print("✓ All models saved!")


if __name__ == "__main__":
    print("ML Model Trainer - Quick Test")
    print("This module should be run from train_models.py")