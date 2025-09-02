class Sentinel2ZarrDataset(Dataset):
    def __init__(self, df_x, res, bands, target_size=(320, 320)):
        self.df_x = df_x
        self.res = res
        self.bands = bands
        self.target_size = target_size
        self.res_key = f"r{res}"
        self.x_res = f"x_{res}"
        self.y_res = f"y_{res}"

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, index):
        zarr_path = self.df_x["path"].iloc[index] + ".zarr"
        datatree = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
        data = datatree.measurements.reflectance[self.res_key]
        data = data.to_dataset()
        data = data[self.bands].to_dataarray()

        # --- Get chunk layout ---
        band  = self.bands[0]
        chunk_size_y = data.chunksizes[self.y_res][0]
        chunk_size_x = data.chunksizes[self.x_res][0]
        nb_chunks_y = len(data.chunksizes[self.y_res])
        nb_chunks_x = len(data.chunksizes[self.x_res])

        all_chunks, all_masks = [], []

        for row in range(nb_chunks_y):  # Y direction
            for col in range(nb_chunks_x):  # X direction
                y_start = row * chunk_size_y
                x_start = col * chunk_size_x
                chunk_ds = data.isel(
                            {self.y_res: slice(y_start, y_start + chunk_size_y),
                            self.x_res: slice(x_start, x_start + chunk_size_x)}
                        )

                chunk_array = chunk_ds.values.astype(np.float32)
                chunk_array, mask_array = normalize(chunk_array)
                # logger.debug(f"[{index}] Chunk ({row},{col}) normalized")

                # Convert to torch [C, H, W]
                chunk_tensor = torch.from_numpy(chunk_array).float()
                mask_tensor = torch.from_numpy(mask_array).float()

                # Resize to target size
                chunk_tensor = F.interpolate(
                    chunk_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='nearest'
                ).squeeze(0)
                mask_tensor = mask_tensor > 0.5

                all_chunks.append(chunk_tensor)
                all_masks.append(mask_tensor)

        chunks_grid = torch.stack(all_chunks).view(nb_chunks_y, nb_chunks_x, *all_chunks[0].shape)
        masks_grid = torch.stack(all_masks).view(nb_chunks_y, nb_chunks_x, *all_masks[0].shape)
        meta = (nb_chunks_y, nb_chunks_x, chunk_size_y, chunk_size_x)

        datatree.close()
        # logger.debug(f"[{index}] Finished processing -> chunks_grid: {chunks_grid.shape}, masks_grid: {masks_grid.shape}")

        return chunks_grid, masks_grid, meta