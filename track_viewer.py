"""
Napari Track Viewer
-------------------
Usage: 
    python track_viewer.py <tif_folder> <csv_file>
    
Example:
    python track_viewer.py tif test_spots_with_seg_info.csv
    python track_viewer.py /path/to/masks /path/to/tracks.csv

This will open a napari viewer where you can click on segmented cells
to visualize their tracks and see detailed information.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from tifffile import imread
import napari
from qtpy import QtWidgets, QtCore, QtGui


# =============================================================================
# DATA LOADING
# =============================================================================
def load_stack_from_folder(folder):
    """
    Load and stack 2D TIF images from a folder.
    Returns (T, Y, X) numpy array.
    """
    # 1) Collect full paths (handles .tif)
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(".tif")]
    
    if not files:
        raise ValueError(f"No .tif files found in folder: {folder}")
    
    # 2) Sort by the trailing number like ...-10000_cp_masks.tif
    def extract_num(p):
        m = re.search(r"-(\d+)(?=_cp_masks\.tif$|\.tif$)", os.path.basename(p), re.IGNORECASE)
        return int(m.group(1)) if m else -1
    
    files = sorted(files, key=extract_num)
    
    # 3) Read each 2D image and stack -> (T, Y, X)
    planes = []
    target_shape = None
    for f in files:
        img = np.squeeze(imread(f))
        if img.ndim != 2:
            raise ValueError(f"{os.path.basename(f)} is not 2D after squeeze, shape={img.shape}")
        if target_shape is None:
            target_shape = img.shape
        elif img.shape != target_shape:
            raise ValueError(f"Shape mismatch: {os.path.basename(f)} has {img.shape}, expected {target_shape}")
        planes.append(img)
    
    stack = np.stack(planes, axis=0)  # (T, Y, X)
    print(f"✓ Loaded stack: shape={stack.shape}, dtype={stack.dtype}, files={len(files)}")
    return stack


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def track_time_span(track_df, t_col="POSITION_T"):
    """
    Compute the time span of a track.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame containing track data
    t_col : str
        Column name for time/frame
        
    Returns
    -------
    str
        Formatted string like "frames 12 → 28"
    """
    if t_col not in track_df.columns or track_df.empty:
        return "frames ? → ?"
    frames = track_df[t_col].dropna().astype(int).to_numpy()
    if frames.size == 0:
        return "frames ? → ?"
    t_min = int(frames.min())
    t_max = int(frames.max())
    return f"frames {t_min} → {t_max}"


# =============================================================================
# GUI WIDGET
# =============================================================================
class TrackInfoWidget(QtWidgets.QWidget):
    """Widget displayed in napari dock showing track information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header labels
        self.header_seg_label = QtWidgets.QLabel("SEG_LABEL: -")
        self.header_frame     = QtWidgets.QLabel("Clicked frame: -")
        self.header_track     = QtWidgets.QLabel("TRACK_ID: -")
        self.header_span      = QtWidgets.QLabel("Track framespan: -")

        # Make some headers bold
        bold_font = self.header_seg_label.font()
        bold_font.setBold(True)
        bold_font.setPointSize(bold_font.pointSize() + 1)
        self.header_seg_label.setFont(bold_font)
        self.header_track.setFont(bold_font)

        layout.addWidget(self.header_seg_label)
        layout.addWidget(self.header_frame)
        layout.addWidget(self.header_track)
        layout.addWidget(self.header_span)

        # Table showing entire track (all frames of that TRACK_ID)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Allow selecting cells (not just whole rows) and multi-select
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

        # --- Copy support (Ctrl+C and context menu) ---
        copy_action = QtWidgets.QAction("Copy", self.table)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selection)
        self.table.addAction(copy_action)

        self.table.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.table.addAction(copy_action)

    def copy_selection(self):
        """Copy the currently selected table block to the clipboard as TSV."""
        ranges = self.table.selectedRanges()
        if not ranges:
            return
        sr = ranges[0]
        rows = range(sr.topRow(), sr.bottomRow() + 1)
        cols = range(sr.leftColumn(), sr.rightColumn() + 1)

        lines = []
        for r in rows:
            row_vals = []
            for c in cols:
                item = self.table.item(r, c)
                row_vals.append(item.text() if item else "")
            lines.append("\t".join(row_vals))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def update_track_view(self, seg_label, t_click, track_id_text, span_text, track_df):
        """
        Update the widget with track information.
        
        Parameters
        ----------
        seg_label : int
            Label value that was clicked
        t_click : int
            Frame number where click occurred
        track_id_text : str
            Track ID as string
        span_text : str
            Formatted time span string
        track_df : pd.DataFrame
            Full track data
        """
        self.header_seg_label.setText(f"SEG_LABEL: {seg_label}")
        self.header_frame.setText(f"Clicked frame: {t_click}")
        self.header_track.setText(f"TRACK_ID: {track_id_text}")
        self.header_span.setText(f"Track framespan: {span_text}")

        # Choose columns to display
        show_cols = [
            "TRACK_ID",
            "POSITION_T",
            "POSITION_Y",
            "POSITION_X",
            "SEG_LABEL",
            "SEG_CY",
            "SEG_CX",
            "AREA",
            "AREA_R",
            "SEG_AREA",
            "SEG_AREA_R",
        ]
        show_cols = [c for c in show_cols if c in track_df.columns]

        self.table.setColumnCount(len(show_cols))
        self.table.setHorizontalHeaderLabels(show_cols)
        self.table.setRowCount(len(track_df))

        for r, (_, row) in enumerate(track_df.iterrows()):
            for c, colname in enumerate(show_cols):
                item = QtWidgets.QTableWidgetItem(str(row[colname]))
                # Not editable
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()


# =============================================================================
# MAIN VIEWER
# =============================================================================
def launch_viewer(stack, df_annot):
    """
    Launch napari viewer with track visualization capabilities.
    
    Parameters
    ----------
    stack : np.ndarray
        (T, Y, X) label stack
    df_annot : pd.DataFrame
        DataFrame with columns: TRACK_ID, POSITION_T, POSITION_Y, POSITION_X,
        SEG_LABEL, and optionally SEG_CY, SEG_CX, SEG_AREA
    """
    print("✓ Launching napari viewer...")
    
    viewer = napari.Viewer()
    labels_layer = viewer.add_labels(stack, name='segmentation')

    # Determine dimensionality
    if stack.ndim == 3:
        ndim_pts = 3
        init_pts = np.empty((0, 3))
    elif stack.ndim == 2:
        ndim_pts = 2
        init_pts = np.empty((0, 2))
    else:
        raise ValueError("stack must be 2D (Y,X) or 3D (T,Y,X).")

    # Two point layers: one for all track points (red), one for clicked point (yellow)
    all_track_points = viewer.add_points(
        init_pts,
        name="track_points",
        face_color='red',
        edge_color='darkred',
        size=6,
        blending='translucent',
        visible=True,
        ndim=ndim_pts,
    )

    highlight_point = viewer.add_points(
        init_pts,
        name="clicked_point",
        face_color='yellow',
        edge_color='black',
        size=10,
        blending='translucent',
        visible=True,
        ndim=ndim_pts,
    )

    # Create and add dock widget
    info_widget = TrackInfoWidget()
    viewer.window.add_dock_widget(
        info_widget,
        name="Track Info",
        area="right"
    )

    # =========================================================================
    # CLICK CALLBACK
    # =========================================================================
    def on_click(layer, event):
        """Handle mouse clicks on the labels layer."""
        # Respond only to left click press
        if event.type != 'mouse_press':
            return
        if event.button != 1:
            return

        # Map world coords -> data coords in the labels layer
        data_pos = layer.world_to_data(event.position)
        data_pos = np.round(data_pos).astype(int)

        # Safety bounds check
        if np.any(data_pos < 0) or np.any(data_pos >= layer.data.shape):
            return

        # Read (t,y,x) and seg_label
        if layer.data.ndim == 3:
            t_click, yy, xx = data_pos
            t_click = int(t_click)
            yy = int(yy)
            xx = int(xx)
            seg_label_val = int(layer.data[t_click, yy, xx])
        else:
            # 2D case
            t_click = 0
            yy = int(data_pos[0])
            xx = int(data_pos[1])
            seg_label_val = int(layer.data[yy, xx])

        if seg_label_val == 0:
            # Background - ignore
            return

        # ---------------------------------------------------------------------
        # Find TRACK_ID at this frame/time
        # ---------------------------------------------------------------------
        rows_here = df_annot[
            (df_annot["SEG_LABEL"] == seg_label_val) &
            (df_annot["POSITION_T"] == t_click)
        ]

        if rows_here.empty or "TRACK_ID" not in rows_here.columns:
            # Couldn't map this label+time to a specific track
            track_id_text = "not found"
            whole_track_df = pd.DataFrame([])
            span_text = "frames ? → ?"
        else:
            # Get track ID (take first if multiple)
            track_ids = (
                rows_here["TRACK_ID"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            if len(track_ids) == 0:
                track_id_text = "None"
                whole_track_df = pd.DataFrame([])
                span_text = "frames ? → ?"
            else:
                track_id_val = int(track_ids[0])
                track_id_text = str(track_id_val)

                # Pull the ENTIRE TRACK across all frames
                whole_track_df = df_annot[
                    df_annot["TRACK_ID"].astype(int) == track_id_val
                ].copy()

                # Sort by time
                if "POSITION_T" in whole_track_df.columns:
                    whole_track_df = whole_track_df.sort_values("POSITION_T")

                # Compute time span
                span_text = track_time_span(whole_track_df, t_col="POSITION_T")

        # ---------------------------------------------------------------------
        # Update dock widget
        # ---------------------------------------------------------------------
        info_widget.update_track_view(
            seg_label=seg_label_val,
            t_click=t_click,
            track_id_text=track_id_text,
            span_text=span_text,
            track_df=whole_track_df,
        )

        # ---------------------------------------------------------------------
        # Display all track points and highlight clicked one
        # ---------------------------------------------------------------------
        if whole_track_df.empty:
            # Clear both point layers
            all_track_points.data = np.empty((0, ndim_pts))
            highlight_point.data = np.empty((0, ndim_pts))
        else:
            # Build coordinates for all points in the track using DataFrame values
            if stack.ndim == 3:
                # Use POSITION_T, POSITION_Y, POSITION_X from DataFrame
                all_coords = whole_track_df[['POSITION_T', 'POSITION_Y', 'POSITION_X']].values
            else:
                # 2D case: just Y, X
                all_coords = whole_track_df[['POSITION_Y', 'POSITION_X']].values
            
            all_track_points.data = all_coords

            # Highlight the clicked point using POSITION_Y, POSITION_X from DataFrame
            clicked_row = rows_here.iloc[0]
            cy = float(clicked_row['POSITION_Y'])
            cx = float(clicked_row['POSITION_X'])
            
            if stack.ndim == 3:
                highlight_point.data = np.array([[t_click, cy, cx]], dtype=float)
            else:
                highlight_point.data = np.array([[cy, cx]], dtype=float)

        # Debug print
        print(
            f"CLICK: seg_label={seg_label_val} @ frame={t_click} "
            f"→ TRACK_ID={track_id_text}, span={span_text}, "
            f"points={len(whole_track_df)}"
        )

    # Attach callback
    labels_layer.mouse_drag_callbacks.append(on_click)

    print("✓ Viewer ready! Click on any cell to see its track.")
    print("  - RED points: all detections in the track")
    print("  - YELLOW point: the clicked detection")
    
    # Run napari
    napari.run()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def print_usage():
    """Print usage information."""
    print("Usage:")
    print("  python track_viewer.py <tif_folder> <csv_file>")
    print()
    print("Example:")
    print("  python track_viewer.py tif test_spots_with_seg_info.csv")
    print("  python track_viewer.py /path/to/masks /path/to/tracks.csv")
    print()


def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    """Main function to load data and launch viewer."""
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Error: Incorrect number of arguments.")
        print()
        print_usage()
        sys.exit(1)
    
    tif_folder = sys.argv[1]
    csv_file = sys.argv[2]
    
    # Check if paths exist
    if not os.path.exists(tif_folder):
        print(f"Error: TIF folder not found: {tif_folder}")
        sys.exit(1)
    
    if not os.path.isdir(tif_folder):
        print(f"Error: Path is not a directory: {tif_folder}")
        sys.exit(1)
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("NAPARI TRACK VIEWER")
    print("=" * 70)
    
    # Load stack
    print(f"\n1. Loading image stack from folder: {tif_folder}")
    try:
        stack = load_stack_from_folder(tif_folder)
    except Exception as e:
        print(f"Error loading stack: {e}")
        sys.exit(1)
    
    # Load annotations
    print(f"\n2. Loading annotations from: {csv_file}")
    try:
        df_annot = pd.read_csv(csv_file)
        print(f"✓ Loaded annotations: {len(df_annot)} rows, {len(df_annot.columns)} columns")
        print(f"  Columns: {', '.join(df_annot.columns.tolist())}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Check for required columns
    required_cols = ["TRACK_ID", "POSITION_T", "POSITION_Y", "POSITION_X", "SEG_LABEL"]
    missing = [col for col in required_cols if col not in df_annot.columns]
    if missing:
        print(f"⚠ Warning: Missing required columns: {missing}")
        print(f"  Required columns: {required_cols}")
    
    # Launch viewer
    print("\n3. Launching napari viewer...")
    try:
        launch_viewer(stack, df_annot)
    except Exception as e:
        print(f"Error launching viewer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
