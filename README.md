# Napari Track Viewer

An interactive napari tool for exploring **cell segmentations + cell tracks** across time.

This viewer allows you to:

- load a folder of sequential segmentation `.tif` files  
- load a CSV file containing object-level annotations  
- click on a segmented cell in napari  
- instantly view its complete tracking data  
- highlight all detections of that track  
- **copy track data directly from the table (Ctrl+C)**

---

## 🚀 Quick Start

### Run the viewer
```
bash
python track_viewer.py <tif_folder> <csv_file>
python track_viewer.py tif test_spots_with_seg_info.csv
```
## 📂 Input Requirements
TIF folder

Contains sequential 2D segmentation mask images

Must contain .tif

Files are loaded and stacked in sorted order

Final shape is (T, Y, X)

## 🧭 How to Use

Click a segmented object in napari

The Track Info panel displays:

segmentation label

clicked frame

track ID

frame range

complete table of detections

Track points are displayed:
- 🔴 red points = entire track

- 🟡 yellow point = clicked detection
