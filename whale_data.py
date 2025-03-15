import ee
import numpy as np
import matplotlib.pyplot as plt

# Initialize Earth Engine
ee.Initialize(project='ee-sudhanvi24')

# Define Norwegian Sea region (whale hotspot)
region = ee.Geometry.Rectangle([15, 70, 20, 75])  # Lon/Lat bounds

# Fetch NOAA AVHRR SST (daily, 1km resolution)
sst_collection = ee.ImageCollection('NOAA/AVHRR_Pathfinder_SST') \
    .filterBounds(region) \
    .filterDate('2024-01-01', '2024-12-31') \
    .select('sst')
sst_image = sst_collection.median()  # Median SST in °C (already scaled)

# Simulate a 1km pixel with base SST (~8°C, Norwegian Sea winter)
pixel_size = 1000  # 1km resolution
base_sst = 8.0    # Typical winter SST

# Fake whale pod (0.2°C anomaly over 100m within 1km pixel)
pixel = base_sst * np.ones((10, 10))  # 10x10 grid = 100m sub-pixels
pixel[4:6, 4:6] = 8.2                 # Whale pod anomaly
whale_avg = np.mean(pixel)            # ~8.04°C

# Fake ship path (no whales)
ship_pixel = base_sst * np.ones((10, 10))
ship_avg = np.mean(ship_pixel)        # ~8.00°C

print(f"Whale Pixel Avg SST: {whale_avg:.2f}°C")
print(f"Ship Path Avg SST: {ship_avg:.2f}°C")

# Visualize
plt.imshow(pixel, cmap='coolwarm', vmin=8, vmax=8.5)
plt.title('Simulated Whale Pod SST (1km Pixel)')
plt.colorbar(label='SST (°C)')
plt.savefig('whale_sst.png')
plt.show()

# Save for ML
import pandas as pd
data = pd.DataFrame({
    'SST': [whale_avg, ship_avg],
    'Label': ['Whale', 'No Whale']
})
data.to_csv('simulated_sst.csv', index=False)