"""
Smart Study Session Optimizer - ML Engine
=======================================

Machine Learning components for:
- Pattern recognition in study habits
- Break timing optimization
- Subject recommendation
- Focus score prediction
"""

import os
import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

class StudyPatternAnalyzer:
    """Analyzes study patterns and provides ML-powered recommendations"""
    
    def __init__(self, db_path="../data/sessions.db", model_path="../data/models/"):
        self.db_path = os.path.abspath(db_path)
        self.model_path = os.path.abspath(model_path)
        os.makedirs(self.model_path, exist_ok=True)
        
        # ML models
        self.break_predictor = None
        self.subject_recommender = None
        self.pattern_clusterer = None
        self.scaler = StandardScaler()
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists(f"{self.model_path}/break_predictor.joblib"):
                self.break_predictor = joblib.load(f"{self.model_path}/break_predictor.joblib")
                print("Loaded break predictor model")
            
            if os.path.exists(f"{self.model_path}/subject_recommender.joblib"):
                self.subject_recommender = joblib.load(f"{self.model_path}/subject_recommender.joblib")
                print("Loaded subject recommender model")
            
            if os.path.exists(f"{self.model_path}/pattern_clusterer.joblib"):
                self.pattern_clusterer = joblib.load(f"{self.model_path}/pattern_clusterer.joblib")
                print("Loaded pattern clusterer model")
                
            if os.path.exists(f"{self.model_path}/scaler.joblib"):
                self.scaler = joblib.load(f"{self.model_path}/scaler.joblib")
                print("Loaded feature scaler")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.break_predictor:
                joblib.dump(self.break_predictor, f"{self.model_path}/break_predictor.joblib")
            
            if self.subject_recommender:
                joblib.dump(self.subject_recommender, f"{self.model_path}/subject_recommender.joblib")
            
            if self.pattern_clusterer:
                joblib.dump(self.pattern_clusterer, f"{self.model_path}/pattern_clusterer.joblib")
            
            joblib.dump(self.scaler, f"{self.model_path}/scaler.joblib")
            
            print("Models saved successfully")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_session_data(self) -> pd.DataFrame:
        """Load session data from SQLite database"""
        if not os.path.exists(self.db_path):
            print("Database not found. Creating sample data...")
            return self._create_sample_data()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT 
                        start_time, end_time, duration_minutes, keystroke_count,
                        mouse_movements, active_windows, primary_subject, focus_score
                    FROM sessions
                    ORDER BY start_time
                """, conn)
            
            if df.empty:
                print("No session data found. Creating sample data...")
                return self._create_sample_data()
                
            # Feature engineering
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['hour_of_day'] = df['start_time'].dt.hour
            df['day_of_week'] = df['start_time'].dt.dayofweek
            df['keystrokes_per_minute'] = df['keystroke_count'] / df['duration_minutes'].clip(lower=1)
            df['mouse_per_minute'] = df['mouse_movements'] / df['duration_minutes'].clip(lower=1)
            
            print(f"Loaded {len(df)} sessions from database")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample training data for initial model development"""
        np.random.seed(42)
        
        # Generate 100 sample sessions
        n_samples = 100
        
        data = []
        subjects = ['coding', 'learning', 'browsing', 'terminal', 'other']
        
        for i in range(n_samples):
            # Random session parameters
            duration = np.random.normal(35, 15)  # Average 35-minute sessions
            duration = max(5, duration)  # Minimum 5 minutes
            
            hour = np.random.randint(8, 22)  # 8 AM to 10 PM
            day_of_week = np.random.randint(0, 7)
            
            # Correlate keystrokes with focus and subject
            base_keystrokes = np.random.normal(120, 40)  # Base keystrokes per minute
            if hour < 10 or hour > 20:  # Early morning or late night
                base_keystrokes *= 0.7
            
            subject = np.random.choice(subjects, p=[0.4, 0.3, 0.15, 0.1, 0.05])
            if subject == 'coding':
                base_keystrokes *= 1.3
            elif subject == 'browsing':
                base_keystrokes *= 0.6
            
            keystroke_count = max(10, base_keystrokes * duration)
            mouse_movements = keystroke_count * np.random.uniform(0.3, 0.8)
            
            # Focus score based on activity and duration
            focus_score = min(1.0, (keystroke_count / duration) / 150)
            focus_score += np.random.normal(0, 0.1)
            focus_score = max(0, min(1, focus_score))
            
            # Start time (last 30 days)
            start_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
            start_time = start_time.replace(hour=hour, minute=np.random.randint(0, 60))
            end_time = start_time + timedelta(minutes=duration)
            
            data.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': duration,
                'keystroke_count': int(keystroke_count),
                'mouse_movements': int(mouse_movements),
                'primary_subject': subject,
                'focus_score': focus_score,
                'hour_of_day': hour,
                'day_of_week': day_of_week,
                'keystrokes_per_minute': keystroke_count / duration,
                'mouse_per_minute': mouse_movements / duration,
                'active_windows': json.dumps({f"{subject}_app": [{"start": start_time.isoformat()}]})
            })
        
        df = pd.DataFrame(data)
        print(f"Created {len(df)} sample sessions for training")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML models"""
        features = []
        
        for _, row in df.iterrows():
            feature_vector = [
                row['duration_minutes'],
                row['keystrokes_per_minute'],
                row['mouse_per_minute'],
                row['focus_score'],
                row['hour_of_day'],
                row['day_of_week'],
                1 if row['primary_subject'] == 'coding' else 0,
                1 if row['primary_subject'] == 'learning' else 0,
                1 if row['primary_subject'] == 'browsing' else 0,
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_break_predictor(self, df: pd.DataFrame):
        """Train model to predict optimal break timing"""
        features = self.extract_features(df)
        
        # Target: next session should be break if current session > 45 min or focus < 0.3
        targets = []
        for _, row in df.iterrows():
            should_break = (row['duration_minutes'] > 45) or (row['focus_score'] < 0.3)
            targets.append(1 if should_break else 0)
        
        targets = np.array(targets)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, targets, test_size=0.2, random_state=42
        )
        
        self.break_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.break_predictor.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.break_predictor.score(X_train, y_train)
        test_score = self.break_predictor.score(X_test, y_test)
        
        print(f"Break predictor trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    def train_pattern_clustering(self, df: pd.DataFrame):
        """Train clustering model to identify study patterns"""
        features = self.extract_features(df)
        features_scaled = self.scaler.fit_transform(features)
        
        # Use KMeans to find study patterns
        self.pattern_clusterer = KMeans(n_clusters=4, random_state=42)
        clusters = self.pattern_clusterer.fit_predict(features_scaled)
        
        # Analyze clusters
        df['cluster'] = clusters
        for i in range(4):
            cluster_data = df[df['cluster'] == i]
            avg_duration = cluster_data['duration_minutes'].mean()
            avg_focus = cluster_data['focus_score'].mean()
            common_subject = cluster_data['primary_subject'].mode().iloc[0] if len(cluster_data) > 0 else 'unknown'
            
            print(f"Cluster {i}: Avg duration={avg_duration:.1f}min, Focus={avg_focus:.2f}, Subject={common_subject}")
    
    def predict_break_recommendation(self, current_session_features: Dict) -> Dict[str, Any]:
        """Predict if user should take a break"""
        if not self.break_predictor:
            return self._fallback_break_recommendation(current_session_features)
        
        try:
            # Convert current session to feature vector
            feature_vector = np.array([[
                current_session_features.get('duration_minutes', 0),
                current_session_features.get('keystrokes_per_minute', 0),
                current_session_features.get('mouse_per_minute', 0),
                current_session_features.get('focus_score', 0.5),
                current_session_features.get('hour_of_day', 12),
                current_session_features.get('day_of_week', 1),
                1 if current_session_features.get('primary_subject') == 'coding' else 0,
                1 if current_session_features.get('primary_subject') == 'learning' else 0,
                1 if current_session_features.get('primary_subject') == 'browsing' else 0,
            ]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict
            break_probability = self.break_predictor.predict(feature_vector_scaled)[0]
            
            if break_probability > 0.7:
                return {
                    "type": "long_break",
                    "duration_minutes": 15,
                    "message": "ML suggests a long break - you've been highly focused!",
                    "confidence": min(0.95, break_probability),
                    "urgency": "high"
                }
            elif break_probability > 0.4:
                return {
                    "type": "short_break", 
                    "duration_minutes": 5,
                    "message": "ML suggests a quick break to maintain focus.",
                    "confidence": break_probability,
                    "urgency": "medium"
                }
            else:
                return {
                    "type": "continue",
                    "duration_minutes": 0,
                    "message": "ML says keep going - you're in a productive state!",
                    "confidence": 1 - break_probability,
                    "urgency": "low"
                }
                
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._fallback_break_recommendation(current_session_features)
    
    def _fallback_break_recommendation(self, features: Dict) -> Dict[str, Any]:
        """Fallback rule-based recommendation"""
        duration = features.get('duration_minutes', 0)
        focus_score = features.get('focus_score', 0.5)
        
        if duration > 50 or focus_score < 0.3:
            return {
                "type": "long_break",
                "duration_minutes": 15, 
                "message": "Time for a break based on session length or focus drop.",
                "confidence": 0.6,
                "urgency": "high"
            }
        elif duration > 25:
            return {
                "type": "short_break",
                "duration_minutes": 5,
                "message": "Consider a short break to recharge.",
                "confidence": 0.5,
                "urgency": "medium"
            }
        else:
            return {
                "type": "continue", 
                "duration_minutes": 0,
                "message": "Keep going - you're doing well!",
                "confidence": 0.6,
                "urgency": "low"
            }
    
    def train_all_models(self):
        """Train all ML models"""
        print("Loading session data...")
        df = self.load_session_data()
        
        if len(df) < 10:
            print("Insufficient data for training. Using sample data.")
            df = self._create_sample_data()
        
        print("Training break predictor...")
        self.train_break_predictor(df)
        
        print("Training pattern clustering...")
        self.train_pattern_clustering(df)
        
        print("Saving models...")
        self._save_models()
        
        print("ML training complete!")


def main():
    """Test ML functionality"""
    print("Smart Study Optimizer - ML Engine Test")
    print("=====================================")
    
    analyzer = StudyPatternAnalyzer()
    
    # Train models
    analyzer.train_all_models()
    
    # Test prediction
    test_features = {
        'duration_minutes': 45,
        'keystrokes_per_minute': 120,
        'mouse_per_minute': 60,
        'focus_score': 0.4,
        'hour_of_day': 14,
        'day_of_week': 2,
        'primary_subject': 'coding'
    }
    
    recommendation = analyzer.predict_break_recommendation(test_features)
    print(f"\nTest Recommendation: {recommendation}")


if __name__ == "__main__":
    main()