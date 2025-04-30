import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

data_files = {
    "DDoS": "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "PortScan": "data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "WebAttacks": "data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "DoS": "data/Wednesday-workingHours.pcap_ISCX.csv",
    "Normal": "data/Monday-WorkingHours.pcap_ISCX.csv"
}

def load_and_preprocess_data(chunk_size=50000):
    processed_chunks = []
    
    for attack_type, file_path in data_files.items():
        print(f"Processing {attack_type}...")
        try:
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
            
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.str.strip()  # Clean column names
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf with NaN
                chunk.dropna(inplace=True)  # Remove missing values
                
                selected_features = [
                    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                    'Fwd Packet Length Max', 'Bwd Packet Length Max',
                    'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s',
                    'Label'
                ]
                chunk = chunk[selected_features]
                
                # Encode Labels (0 for BENIGN, 1 for Attack)
                chunk['Label'] = chunk['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
                
                for col in chunk.columns[:-1]:
                    chunk[col] = np.clip(chunk[col], None, np.percentile(chunk[col], 99))
                
                scaler = StandardScaler()
                feature_cols = chunk.columns[:-1]  # All columns except 'Label'
                chunk[feature_cols] = scaler.fit_transform(chunk[feature_cols])
                
                processed_chunks.append(chunk)
        
        except Exception as e:
            print(f"Error processing {attack_type}: {e}")
    
    # Combine all processed chunks
    if processed_chunks:
        full_dataset = pd.concat(processed_chunks, ignore_index=True)
    else:
        raise ValueError("No valid data processed. Check dataset files.")
    
    # Balance dataset (equal attack & benign samples)
    benign_samples = full_dataset[full_dataset['Label'] == 0]
    attack_samples = full_dataset[full_dataset['Label'] == 1]
    
    benign_downsampled = resample(benign_samples, replace=False, 
                                  n_samples=min(len(benign_samples), len(attack_samples)), random_state=42)
    balanced_data = pd.concat([benign_downsampled, attack_samples])
    
    print("Final dataset shape:", balanced_data.shape)
    return balanced_data

if __name__ == "__main__":
    data = load_and_preprocess_data()
    data.to_csv("data/processed_data.csv", index=False)
    print(" Processed dataset saved!")
