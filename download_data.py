from tdc.multi_pred import DDI
import pandas as pd

print("Downloading DrugBank DDI dataset from TDC...")
print("This may take 2-3 minutes on first download...")

try:
    # Load DrugBank DDI dataset
    data = DDI(name='DrugBank')
    df = data.get_data()

    print(f"✓ Successfully loaded {len(df)} drug-drug interactions")
    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")

    # Save to CSV for backup
    df.to_csv('drugbank_ddi_data.csv', index=False)
    print("✓ Data saved to drugbank_ddi_data.csv")

    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())

except Exception as e:
    print(f"✗ ERROR: {e}")
    print("If download failed, check internet connection and try again")
