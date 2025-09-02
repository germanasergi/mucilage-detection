def resample_to_10m(ds, band, ref, folder):
    """
    Resample band to match the resolution & grid of reference band.
    ds: opened .zarr datatree
    band: name of band to resample (string)
    ref: name of reference band (string)
    """
    crs_code = "EPSG:32632"

    # Define reference band
    ref_band = ds[f"measurements/reflectance/r10m/{ref}"]  # reference (10m red)
    ref_band = ref_band.rio.write_crs(crs_code, inplace=True)

    # Band to convert
    if folder == 'measurements':
        band_20m = ds[f"measurements/reflectance/r20m/{band}"]
    else:
        band_20m = ds[f"conditions/mask/l2a_classification/r20m/{band}"] # for classification band
    band_10m = band_20m.rio.write_crs(crs_code, inplace=True)  # ensure CRS

    return band_10m.rio.reproject_match(ref_band)

def compute_amei(ds, eps=1e-6):
    red   = ds["measurements/reflectance/r10m/b04"] / 10000.0
    green = ds["measurements/reflectance/r10m/b03"] / 10000.0
    nir   = resample_to_10m(ds, 'b8a', 'b04', folder='measurements') / 10000.0
    swir  = resample_to_10m(ds, 'b11', 'b04', folder='measurements') / 10000.0

    # AMEI = (2*red + nir - 2*swir) / (green + 0.25*swir)
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    return amei.rename("amei")  # keep DataArray with name


def compute_ndwi(ds, eps=1e-6):
    green = ds["measurements/reflectance/r10m/b03"] / 10000.0
    nir   = resample_to_10m(ds, 'b8a', 'b04', folder='measurements') / 10000.0

    # NDWI = (green - nir) / (green + nir)
    ndwi = (green - nir) / (green + nir + eps)

    return ndwi.rename("ndwi")  # keep DataArray with name


def clean_water_mask(water_mask):
    """
    Fix water mask by:
    1. Keeping only the sea (remove lakes/rivers).
    2. Filling cloud holes in the sea.
    """
    H, W = water_mask.shape
    
    # Keep only the largest connected component that touches border (most prob the sea)
    st = generate_binary_structure(2, 2)   # 8-connectivity
    lab, nlab = label(water_mask, structure=st) # different label for each connected water body
    
    if nlab == 0:
        return np.zeros_like(water_mask, dtype=bool)

    # Find component sizes
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background, we don't consider it

    # Keep the largest component
    largest_label = sizes.argmax()
    sea_only = (lab == largest_label)

    # Fill holes inside sea (caused by clouds)
    sea_filled = binary_fill_holes(sea_only)

    return sea_filled.astype(bool)


def build_stack_10m(ds, bands):
    """
    Return a lazy dask-backed stack (H, W, C) instead of full NumPy.
    """
    stack = []

    for b in bands:
        if b in ds['measurements/reflectance/r10m']:
            arr = ds['measurements/reflectance/r10m'][b] / 10000.0
        elif b in ds['measurements/reflectance/r20m']:
            arr = resample_to_10m(ds, b, 'b04', folder='measurements') / 10000.0
        elif b == "amei":
            arr = compute_amei(ds)
        elif b == "ndwi":
            arr = compute_ndwi(ds)
        else:
            raise ValueError(f"Band {b} not found or not supported.")

        # Expand dims and assign band coordinate for all arrays
        arr = arr.expand_dims(band=[b])
        stack.append(arr)

    # Concatenate along band dimension lazily
    stack = xr.concat(stack, dim="band").transpose("y", "x", "band")
    return stack


def sample_patch_centers(water_mask, n_patches, patch_size=256, border_weight=0.6):
    """
    Randomly sample patch centers with more focus on water-land borders.
    """
    H, W = water_mask.shape
    
    # Create a mask from the shoreline (water/land border) expanding seawards
    border_mask = water_mask & ~binary_erosion(water_mask, iterations=300) # choose pixels number
    
    # Probability map
    probs = np.zeros_like(water_mask, dtype=np.float32)
    probs[border_mask] = border_weight   # 70% on border
    probs[water_mask]  = 1.0 - border_weight  # 30% on water
    
    probs = probs / probs.sum()  # normalize to 1
    
    # Flatten and sample indices
    flat_idx = np.random.choice(H*W, size=n_patches, replace=False, p=probs.ravel()) # randomly picks pixel indices based on the prob map (p)
    centers = np.column_stack(np.unravel_index(flat_idx, (H, W))) # convert indices back into (row, col) coordinates in the 2D image
    
    # Convert to (row, col) top-left corners
    half = patch_size // 2 # centering
    corners = [(max(0, r-half), max(0, c-half)) for r, c in centers]
    del probs
    
    return corners


def extract_patches_focus(stack, water_mask, n_patches=150, patch_size=256):
    """
    Extract (256,256,C) patches with sampling biased toward border regions.
    """
    H, W, C = stack.shape
    corners = sample_patch_centers(water_mask, n_patches, patch_size)
    
    patches = []
    for i, j in corners:
        if i+patch_size <= H and j+patch_size <= W:
            # This loads only the patch into memory (not the whole image)
            patch = stack.isel(
                y=slice(i, i+patch_size),
                x=slice(j, j+patch_size)
            ).to_numpy()
            patches.append(patch)
    del corners

    return patches


def process_folder_opt(zarr_files, bands, water_mask):
    """
    Loop through all zarr files in a folder and extract patches.
    """
    all_patches = []

    for zf in zarr_files:
        print(f"Processing {zf} ...")
        ds = xr.open_datatree(zf, engine="zarr", mask_and_scale=False, chunks={})

        stack = build_stack_10m(ds, bands)
        patches = extract_patches_focus(stack, water_mask)
        
        all_patches.extend(patches)
        del ds, stack, patches # to avoid RAM overload
        gc.collect()
    
    return all_patches