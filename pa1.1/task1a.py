import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union

# Import VGG16_CIFAR base class for reusability
from utils import VGG16_CIFAR


class VGG16_Pruning(VGG16_CIFAR):
    """
    Extends VGG16_CIFAR with unstructured magnitude-based pruning capabilities.

    Inherits all profiling and dataset methods from VGG16_CIFAR.
    Adds pruning-specific methods for Task 1a.
    """

    def __init__(self, num_classes: int, is_pruned: bool = False):
        super().__init__(num_classes, is_pruned)

        self.model: nn.Module

        # Storage for pruning masks (will be populated during pruning)
        self.pruning_masks = {}

    def get_pruneable_layers(self):
        """
        Get all layers that can be pruned (Conv2d and Linear layers).

        Returns:
            list: List of tuples [(layer_name, layer_module), ...]
        """
        pruneable_layers: list = []

        # Iterate through all named modules in the model
        # named_modules() returns generator of (name, module) pairs
        # Example: ('features.0', Conv2d(...)), ('features.1', BatchNorm2d(...)), etc.
        for name, module in self.model.named_modules():
            # Check if module is Conv2d or Linear layer
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                pruneable_layers.append((name, module))

        return pruneable_layers

    def prune_layer_by_magnitude(self, layer: Union[nn.Conv2d, nn.Linear],
                                  sparsity_ratio: float) -> torch.Tensor:
        """
        Prune a layer using magnitude-based pruning (unstructured).

        Mathematical Process:
        1. Get all weights W from the layer (shape varies: Conv2d has 4D, Linear has 2D)
        2. Flatten weights to 1D vector: w = [w₁, w₂, ..., wₙ]
        3. Compute absolute values: |w| = [|w₁|, |w₂|, ..., |wₙ|]
        4. Sort by magnitude and find threshold at k-th percentile where k = sparsity_ratio × 100
        5. Create binary mask: mask[i] = 1 if |wᵢ| >= threshold, else 0
        6. Apply mask: W_pruned = W ⊙ mask (element-wise multiplication)

        Intuition:
        - Weights with small magnitudes contribute less to the output
        - Setting them to zero has minimal impact on accuracy
        - This is "unstructured" because we prune individual weights, not entire channels/filters

        Parameters:
            layer (nn.Conv2d or nn.Linear): Layer to prune
            sparsity_ratio (float): Fraction of weights to prune (0.0 to 1.0)

        Returns:
            torch.Tensor: Binary mask (1 = keep, 0 = prune)
        """
        # Step 1: Get the weight tensor from the layer
        # For Conv2d: shape is (out_channels, in_channels, kernel_h, kernel_w)
        # For Linear: shape is (out_features, in_features)
        weight_tensor = layer.weight.data

        # Store original device for later
        original_device = weight_tensor.device

        # Step 2: Flatten the weight tensor to 1D for easier processing
        # This converts multi-dimensional tensor to a single vector
        # Example: (64, 3, 3, 3) Conv2d → (1728,) flattened vector
        weight_flattened = weight_tensor.flatten()

        # Step 3: Compute absolute values (magnitude) of all weights
        # We use absolute value because both large positive and large negative weights are important
        # |w| tells us the "strength" of the connection, regardless of direction
        weight_magnitudes = torch.abs(weight_flattened)

        # Step 4: Calculate the number of weights to keep (not prune)
        # If sparsity = 0.7, we prune 70% and keep 30%
        total_weights = weight_flattened.numel()  # Total number of weights in the layer

        # Calculate how many weights to keep (not prune) based on sparsity_ratio
        # Formula: num_weights_to_keep = int(total_weights * (1.0 - sparsity_ratio))
        # Explanation:
        # - sparsity_ratio is the fraction of weights to prune (e.g., 0.7 means prune 70%)
        # - (1.0 - sparsity_ratio) gives the fraction to keep (e.g., 1.0 - 0.7 = 0.3, so keep 30%)
        # - We multiply total_weights by this fraction to get the number of weights to keep
        num_weights_to_keep = int(total_weights * (1.0 - sparsity_ratio))

        # Step 5: Find the threshold value (k-th smallest magnitude)
        # torch.kthvalue finds the k-th smallest value in the tensor
        # NOTE: kthvalue doesn't support MPS, so we move to CPU for MPS devices only
        # All weights with magnitude < threshold will be pruned
        # All weights with magnitude >= threshold will be kept
        if num_weights_to_keep > 0:
            # kthvalue returns (value, index) tuple, we only need the value
            # We want the (total - keep + 1)-th smallest value as our threshold
            # Example: 1000 weights, keep 300 → find 701st smallest (prune 700 smallest)

            # Move to CPU only for MPS devices (kthvalue not supported on MPS)
            if original_device.type == 'mps':
                weight_magnitudes_for_kth = weight_magnitudes.cpu()
            else:
                weight_magnitudes_for_kth = weight_magnitudes

            threshold_value = torch.kthvalue(weight_magnitudes_for_kth, total_weights - num_weights_to_keep + 1).values.item()

        # Step 6: Create binary pruning mask
        # mask[i] = 1 if |weight[i]| >= threshold (keep the weight)
        # mask[i] = 0 if |weight[i]| < threshold (prune the weight)
        # We use >= to handle ties (weights with exactly threshold magnitude are kept)
        pruning_mask = (torch.abs(weight_tensor) >= threshold_value).float()

        # The mask has the same shape as the original weight tensor
        # This allows element-wise multiplication: pruned_weights = weights * mask
        return pruning_mask

    def apply_mask_to_layer(self, layer: Union[nn.Conv2d, nn.Linear], mask: torch.Tensor):
        """
        Apply pruning mask to a layer's weights (force pruned weights to zero).

        Parameters:
            layer (nn.Conv2d or nn.Linear): Layer to apply mask to
            mask (torch.Tensor): Binary mask (1 = keep, 0 = prune)
        """
        # element-wise multiplication (Hadamard product)
        layer.weight.data = torch.mul(layer.weight.data, mask)

    def perform_sensitivity_analysis(self, data_loader: DataLoader, device: str,
                                     sparsity_levels: list, base_path: str):
        """
        Perform comprehensive sensitivity analysis for all pruneable layers.

        Process:
        For each layer L in the model:
            For each sparsity level S in {0%, 10%, 20%, 50%, 70%, 80%, 90%}:
                1. Create a copy of the baseline model
                2. Prune only layer L at sparsity S
                3. Evaluate Top-1 accuracy
                4. Record accuracy for this (layer, sparsity) combination

        Output: A dictionary mapping each layer to its sensitivity curve

        Sensitivity Curve:
        - X-axis: Sparsity levels (0% to 90%)
        - Y-axis: Top-1 accuracy (%)
        - Interpretation: Steep drop = sensitive layer, flat curve = robust layer

        Parameters:
            data_loader (DataLoader): Test data loader
            device (str): Device to run on ('cpu', 'cuda', 'mps')
            sparsity_levels (list): List of sparsity ratios to test (e.g., [0.0, 0.1, 0.2, ...])
            base_path (str): Base path for saving results

        Returns:
            tuple: (sensitivity_results_dict, baseline_accuracy)
            - sensitivity_results_dict: {layer_name: [acc_at_0%, acc_at_10%, ..., acc_at_90%]}
            - baseline_accuracy: The unpruned model's Top-1 accuracy
        """
        import copy
        import pickle

        # Create save directory
        model_name = "cifar10" if self.num_classes == 10 else "cifar100"
        save_dir = os.path.join(base_path, 'task1a', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)

        # Define save path for this model's results
        save_file = os.path.join(save_dir, f'{model_name}_sensitivity_results.pkl')

        # Try to load existing results
        sensitivity_results = {}
        baseline_accuracy = None
        completed_layers = set()

        if os.path.exists(save_file):
            print(f"\n{'='*80}")
            print(f"FOUND SAVED SENSITIVITY ANALYSIS RESULTS")
            print(f"{'='*80}")
            print(f"Loading from: {save_file}")

            with open(save_file, 'rb') as f:
                saved_data = pickle.load(f)

            sensitivity_results = saved_data.get('sensitivity_results', {})
            baseline_accuracy = saved_data.get('baseline_accuracy')
            saved_sparsity_levels = saved_data.get('sparsity_levels', [])
            completed_layers = set(sensitivity_results.keys())

            print(f"✓ Loaded results for {len(sensitivity_results)} layers")
            if baseline_accuracy:
                print(f"✓ Baseline accuracy: {baseline_accuracy:.2f}%")
            print(f"✓ Sparsity levels: {[f'{s*100:.0f}%' for s in saved_sparsity_levels]}")
            print(f"{'='*80}\n")

        # Get all layers that can be pruned (Conv2d and Linear layers)
        pruneable_layers = self.get_pruneable_layers()

        # Check if all layers are already completed
        all_layer_names = set(name for name, _ in pruneable_layers)
        if completed_layers == all_layer_names and baseline_accuracy is not None:
            print("✓ All layers already analyzed. Returning cached results.\n")
            return sensitivity_results, baseline_accuracy

        # Evaluate baseline accuracy if not already done
        if baseline_accuracy is None:
            print("Evaluating baseline model (0% sparsity)...")
            baseline_accuracy, _ = self.profile_accuracy(data_loader, device)
            print(f"Baseline Top-1 Accuracy: {baseline_accuracy:.2f}%\n")

            # Save baseline immediately
            with open(save_file, 'wb') as f:
                pickle.dump({
                    'sensitivity_results': sensitivity_results,
                    'baseline_accuracy': baseline_accuracy,
                    'sparsity_levels': sparsity_levels,
                    'num_classes': self.num_classes
                }, f)

        # Iterate through each pruneable layer
        remaining_layers = [(name, layer) for name, layer in pruneable_layers if name not in completed_layers]

        if remaining_layers:
            print(f"Starting sensitivity analysis...")
            print(f"  Total layers: {len(pruneable_layers)}")
            print(f"  Completed: {len(completed_layers)}")
            print(f"  Remaining: {len(remaining_layers)}")
            print(f"Testing sparsity levels: {[int(s*100) for s in sparsity_levels]}%\n")
        else:
            print("✓ All layers already analyzed.\n")

        for layer_idx, (layer_name, layer_module) in enumerate(remaining_layers):
            # Calculate actual progress
            total_layers = len(pruneable_layers)
            current_layer_num = len(completed_layers) + layer_idx + 1
            print(f"[{current_layer_num}/{total_layers}] Analyzing layer: {layer_name}")

            # Get layer shape and parameter count for informative output
            num_params = layer_module.weight.numel()
            layer_shape = tuple(layer_module.weight.shape)
            print(f"  Shape: {layer_shape}, Parameters: {num_params:,}")

            # Store accuracy at each sparsity level for this layer
            layer_accuracies = []

            # Test each sparsity level
            for sparsity in sparsity_levels:
                sparsity_percent = int(sparsity * 100)

                # Create a deep copy of the model to avoid modifying the original
                # Type annotation helps Pylance understand the return type
                model_copy: nn.Module = copy.deepcopy(self.model)

                # Get the layer from the copied model
                layer_to_prune: Union[nn.Conv2d, nn.Linear, nn.Module, None] = None
                for name, module in model_copy.named_modules():
                    if name == layer_name:
                        layer_to_prune = module
                        break

                # Verify layer was found and is correct type
                if not isinstance(layer_to_prune, (nn.Conv2d, nn.Linear)):
                    raise TypeError(f"Layer {layer_name} is not Conv2d or Linear")

                # Prune the layer
                mask = self.prune_layer_by_magnitude(layer_to_prune, sparsity)
                self.apply_mask_to_layer(layer_to_prune, mask)

                # Evaluate accuracy with the pruned layer
                # Temporarily replace the model
                original_model = self.model
                self.model = model_copy
                self.model.eval()

                accuracy, _ = self.profile_accuracy(data_loader, device)

                # Restore original model
                self.model = original_model

                layer_accuracies.append(accuracy)

                # Calculate accuracy drop from baseline
                accuracy_drop = baseline_accuracy - accuracy

                print(f"    Sparsity {sparsity_percent:2d}%: Accuracy = {accuracy:.2f}%, Drop = {accuracy_drop:+.2f}%")

            # Store results for this layer
            sensitivity_results[layer_name] = layer_accuracies

            # Save immediately after each layer completes
            with open(save_file, 'wb') as f:
                pickle.dump({
                    'sensitivity_results': sensitivity_results,
                    'baseline_accuracy': baseline_accuracy,
                    'sparsity_levels': sparsity_levels,
                    'num_classes': self.num_classes
                }, f)
            print(f"  ✓ Layer results saved")
            print()  # Blank line for readability

        if remaining_layers:
            print(f"✓ All layers completed. Final results saved to: {save_file}\n")

        return sensitivity_results, baseline_accuracy

    def plot_sensitivity_curves(self, sensitivity_results: dict, sparsity_levels: list,
                               baseline_accuracy: float, base_path: str):
        """
        Visualize sensitivity analysis with accuracy percentages shown in legend.

        Parameters:
            sensitivity_results (dict): Results from perform_sensitivity_analysis
            sparsity_levels (list): Sparsity levels tested
            baseline_accuracy (float): Baseline accuracy for reference
            model_name (str): Name of model for plot title (optional, auto-detected from num_classes)
        """

        # Auto-detect model name from num_classes if not provided
        model_name = "CIFAR-10" if self.num_classes == 10 else "CIFAR-100"

        sparsity_percentages = [s * 100 for s in sparsity_levels]

        plt.figure(figsize=(14, 9))

        # Plot a curve for each layer
        for layer_name, accuracies in sensitivity_results.items():
            # Get the final accuracy (at highest sparsity, typically 90%)
            final_accuracy = accuracies[-1]

            # Create label with layer name and final accuracy
            label = f'{layer_name}: {final_accuracy:.1f}%'

            plt.plot(sparsity_percentages, accuracies, marker='o',
                    label=label, linewidth=2, markersize=6, alpha=0.8)

        # Add baseline accuracy
        plt.axhline(y=baseline_accuracy, color='black', linestyle='--', linewidth=2.5,
                    label=f'Baseline (No Pruning): {baseline_accuracy:.2f}%')

        plt.xlabel('Sparsity Level (%)', fontsize=13, fontweight='bold')
        plt.ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
        plt.title(f'Layer-wise Sensitivity Analysis: {model_name}', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')

        # Adjust legend to fit in the box nicely
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8.5,
                  framealpha=0.95, edgecolor='black')

        plt.tight_layout()

        # Save the plot
        save_dir = f'{base_path}/task1a'
        os.makedirs(save_dir, exist_ok=True)

        # Create filename from model name
        # CIFAR-10 -> cifar10, CIFAR-100 -> cifar100
        filename = f"{model_name.lower().replace('-', '')}_sensitivity_analysis.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Sensitivity curves saved to: {save_path}")

        plt.show()


    def print_sensitivity_summary(self, sensitivity_results: dict, sparsity_levels: list,
                                 baseline_accuracy: float):
        """
        Print a summary table of sensitivity analysis results.

        Parameters:
            sensitivity_results (dict): Results from perform_sensitivity_analysis
            sparsity_levels (list): Sparsity levels tested
            baseline_accuracy (float): Baseline accuracy for reference
        """
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nBaseline Top-1 Accuracy: {baseline_accuracy:.2f}%\n")

        # Print header
        header = f"{'Layer':<20s} | "
        for sparsity in sparsity_levels:
            header += f"{sparsity*100:4.0f}% | "
        print(header)
        print("-" * len(header))

        # Print each layer's results
        for layer_name, accuracies in sensitivity_results.items():
            row = f"{layer_name:<20s} | "
            for accuracy in accuracies:
                drop = baseline_accuracy - accuracy
                row += f"{drop:+5.1f} | "
            print(row)

        print("\n(Values show accuracy drop from baseline in percentage points)")
        print("="*80)

    def calculate_overall_sparsity(self, layer_sparsity_dict: dict) -> tuple:
        """
        Calculate the overall sparsity ratio across all specified layers.

        Formula:
        Overall Sparsity = (Total Pruned Parameters) / (Total Parameters)

        Where:
        - Total Pruned Parameters = Σ (sparsity_i × params_i) for each layer i
        - Total Parameters = Σ params_i for all pruneable layers

        Parameters:
            layer_sparsity_dict (dict): Dictionary mapping layer names to sparsity ratios

        Returns:
            tuple: (overall_sparsity, total_params, total_pruned_params)
        """
        total_params = 0
        total_pruned_params = 0

        # Iterate through all specified layers
        for layer_name, sparsity in layer_sparsity_dict.items():
            # Find the layer in the model
            layer_module = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer_module = module
                    break

            if layer_module is None or not isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                continue

            # Calculate number of parameters in this layer
            num_params = layer_module.weight.numel()
            total_params += num_params

            # Calculate pruned parameters based on sparsity ratio
            pruned_params = int(num_params * sparsity)
            total_pruned_params += pruned_params

        # Calculate overall sparsity as percentage
        overall_sparsity = total_pruned_params / total_params if total_params > 0 else 0.0

        return overall_sparsity, total_params, total_pruned_params

    def apply_layerwise_pruning(self, layer_sparsity_dict: dict):
        """
        Apply magnitude-based pruning to all layers according to their assigned sparsity ratios.

        Process:
        1. For each layer in the sparsity dictionary
        2. Generate pruning mask based on magnitude
        3. Apply mask to zero out smallest weights
        4. Store mask for later use (needed for saving/loading)

        Parameters:
            layer_sparsity_dict (dict): Dictionary mapping layer names to sparsity ratios

        Returns:
            dict: Dictionary of pruning masks for each layer
        """
        pruning_masks = {}

        print("\n" + "="*80)
        print("APPLYING LAYERWISE PRUNING")
        print("="*80 + "\n")

        # Iterate through all layers in the sparsity dictionary
        for layer_name, sparsity_ratio in layer_sparsity_dict.items():
            # Find the layer in the model
            layer_module = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer_module = module
                    break

            if layer_module is None or not isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                print(f"Warning: Layer {layer_name} not found or not pruneable")
                continue

            # Get layer info
            num_params = layer_module.weight.numel()
            layer_shape = tuple(layer_module.weight.shape)

            # Generate pruning mask
            mask = self.prune_layer_by_magnitude(layer_module, sparsity_ratio)

            # Apply mask to layer
            self.apply_mask_to_layer(layer_module, mask)

            # Store mask
            pruning_masks[layer_name] = mask

            # Calculate actual sparsity achieved
            num_zeros = (mask == 0).sum().item()
            actual_sparsity = num_zeros / num_params

            print(f"{layer_name}:")
            print(f"  Shape: {layer_shape}")
            print(f"  Parameters: {num_params:,}")
            print(f"  Target Sparsity: {sparsity_ratio*100:.1f}%")
            print(f"  Actual Sparsity: {actual_sparsity*100:.1f}%")
            print()

        # Store masks in the model
        self.pruning_masks = pruning_masks

        print("✓ Layerwise pruning completed\n")
        print("="*80 + "\n")

        return pruning_masks

    def create_sparsity_callback(self, pruning_masks: dict, device: str):
        """
        Create a callback function that enforces sparsity after each optimizer step.

        This is a helper method that creates a callback to be used with the base
        fine_tune() method. The callback re-applies pruning masks to maintain
        sparsity during training.

        Why re-apply masks?
        - Optimizer updates ALL weights (including pruned ones)
        - We must force pruned weights back to zero after each update
        - This maintains the sparsity structure we created

        Parameters:
            pruning_masks (dict): Dictionary of pruning masks {layer_name: mask_tensor}
            device (str): Device to run on ('cpu', 'cuda', 'mps')

        Returns:
            callable: Callback function with signature callback(model)

        """
        def sparsity_callback(model: nn.Module):
            """Re-apply pruning masks to maintain sparsity."""
            with torch.no_grad():
                for layer_name, layer_module in model.named_modules():
                    if layer_name in pruning_masks:
                        if isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                            # Ensure mask is on the same device
                            mask = pruning_masks[layer_name].to(device)
                            layer_module.weight.data.mul_(mask)

        return sparsity_callback

    def convert_to_coo_sparse(self, pruning_masks: dict):
        """
        Convert pruned model weights to COO (Coordinate) sparse format.

        COO Format Explanation:
        - Stores only non-zero values and their coordinates
        - Format: (indices, values, shape)
        - Memory efficient for sparse matrices

        Parameters:
            pruning_masks (dict): Dictionary of pruning masks

        Returns:
            tuple: (sparse_tensors_dict, memory_stats_dict)
        """
        sparse_tensors = {}
        memory_stats = {
            'dense_size_mb': 0.0,
            'sparse_size_mb': 0.0,
            'layers': {}
        }

        print("\n" + "="*80)
        print("CONVERTING TO COO SPARSE FORMAT")
        print("="*80 + "\n")

        for layer_name, layer_module in self.model.named_modules():
            if layer_name in pruning_masks:
                if isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                    # Get the pruned weight tensor
                    weight_tensor = layer_module.weight.data

                    # Calculate dense memory size
                    dense_size_bytes = weight_tensor.numel() * weight_tensor.element_size()
                    dense_size_mb = dense_size_bytes / (1024 * 1024)

                    # Convert to COO sparse format
                    # to_sparse() automatically creates COO format
                    sparse_tensor = weight_tensor.to_sparse()

                    # Calculate sparse memory size
                    # COO format stores: values (float32) + indices (int64 × ndim)
                    num_nonzeros = sparse_tensor._nnz()
                    ndim = sparse_tensor.ndim
                    values_size_bytes = num_nonzeros * 4  # float32 = 4 bytes
                    indices_size_bytes = num_nonzeros * ndim * 8  # int64 = 8 bytes
                    sparse_size_bytes = values_size_bytes + indices_size_bytes
                    sparse_size_mb = sparse_size_bytes / (1024 * 1024)

                    # Store sparse tensor
                    sparse_tensors[layer_name] = sparse_tensor

                    # Update memory stats
                    memory_stats['dense_size_mb'] += dense_size_mb
                    memory_stats['sparse_size_mb'] += sparse_size_mb
                    memory_stats['layers'][layer_name] = {
                        'dense_mb': dense_size_mb,
                        'sparse_mb': sparse_size_mb,
                        'num_nonzeros': num_nonzeros,
                        'total_params': weight_tensor.numel(),
                        'sparsity': 1.0 - (num_nonzeros / weight_tensor.numel())
                    }

                    print(f"{layer_name}:")
                    print(f"  Total params: {weight_tensor.numel():,}")
                    print(f"  Non-zeros: {num_nonzeros:,}")
                    print(f"  Sparsity: {memory_stats['layers'][layer_name]['sparsity']*100:.2f}%")
                    print(f"  Dense size: {dense_size_mb:.2f} MB")
                    print(f"  Sparse size: {sparse_size_mb:.2f} MB")
                    print(f"  Compression: {dense_size_mb/sparse_size_mb:.2f}x")
                    print()

        # Calculate overall compression ratio
        compression_ratio = memory_stats['dense_size_mb'] / memory_stats['sparse_size_mb'] if memory_stats['sparse_size_mb'] > 0 else 0

        print("="*80)
        print("OVERALL SPARSE CONVERSION SUMMARY")
        print("="*80)
        print(f"Total Dense Size:   {memory_stats['dense_size_mb']:.2f} MB")
        print(f"Total Sparse Size:  {memory_stats['sparse_size_mb']:.2f} MB")
        print(f"Compression Ratio:  {compression_ratio:.2f}x")
        print(f"Memory Saved:       {memory_stats['dense_size_mb'] - memory_stats['sparse_size_mb']:.2f} MB")
        print("="*80 + "\n")

        return sparse_tensors, memory_stats

    def verify_mask_coo_consistency(self, pruning_masks: dict, sparse_tensors: dict):
        """
        Verify that pruning masks match COO sparse representation.

        This ensures that the COO conversion correctly preserved the sparsity pattern.

        Parameters:
            pruning_masks (dict): Original binary masks
            sparse_tensors (dict): COO sparse tensors

        Returns:
            dict: Verification results for each layer
        """
        verification_results = {}

        print("\n" + "="*80)
        print("VERIFYING MASK-COO CONSISTENCY")
        print("="*80 + "\n")

        all_consistent = True

        for layer_name in pruning_masks.keys():
            if layer_name in sparse_tensors:
                # Get original mask
                original_mask = pruning_masks[layer_name]

                # Convert sparse tensor back to dense
                sparse_tensor = sparse_tensors[layer_name]
                dense_from_sparse = sparse_tensor.to_dense()

                # Create mask from sparse tensor (non-zero locations)
                mask_from_sparse = (dense_from_sparse != 0).float()

                # Compare masks
                masks_match = torch.all(original_mask == mask_from_sparse).item()

                # Count mismatches if any
                num_mismatches = 0
                if not masks_match:
                    num_mismatches = torch.sum(original_mask != mask_from_sparse).item()
                    all_consistent = False

                verification_results[layer_name] = {
                    'consistent': masks_match,
                    'num_mismatches': num_mismatches,
                    'total_elements': original_mask.numel()
                }

                # Print result
                status = "✓ CONSISTENT" if masks_match else "✗ MISMATCH"
                print(f"{layer_name}: {status}")
                if not masks_match:
                    print(f"  Mismatches: {num_mismatches} / {original_mask.numel()}")

        print("\n" + "="*80)
        if all_consistent:
            print("✓ ALL LAYERS CONSISTENT - COO conversion successful!")
        else:
            print("✗ INCONSISTENCIES FOUND - Please check the conversion")
        print("="*80 + "\n")

        return verification_results

    def replace_weights_with_sparse(self, sparse_tensors: dict):
        """
        Replace model's dense weight tensors with sparse tensors.

        This prepares the model for sparse inference by storing weights in COO format.
        The forward hooks will convert sparse to dense as needed during inference.

        Parameters:
            sparse_tensors (dict): Dictionary of sparse weight tensors

        Returns:
            None (modifies model in-place)
        """
        print("\n" + "="*80)
        print("REPLACING DENSE WEIGHTS WITH SPARSE TENSORS")
        print("="*80 + "\n")

        for layer_name, layer_module in self.model.named_modules():
            if layer_name in sparse_tensors:
                if isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                    # Store weight as sparse tensor (COO format)
                    sparse_weight = sparse_tensors[layer_name]

                    # Replace the weight parameter with sparse tensor
                    # Keep it as sparse - forward hooks will handle conversion
                    # Use nn.Parameter to maintain gradient tracking if needed
                    layer_module.weight = nn.Parameter(sparse_weight, requires_grad=False)

                    print(f"✓ {layer_name}: Replaced with sparse tensor (COO format)")

        print("\n" + "="*80)
        print("✓ All weights replaced with sparse tensors")
        print("="*80 + "\n")

    def setup_sparse_inference_hooks(self, sparse_tensors: dict):
        """
        Setup forward hooks to handle sparse inference.

        For Conv2d layers:
            - Sparse weights are converted to dense before convolution
            - No performance benefit (sparse → dense conversion overhead)

        For Linear layers:
            - Use torch.sparse.mm() for sparse matrix multiplication
            - Potential performance benefit for high sparsity

        Returns:
            list: List of hook handles (for cleanup if needed)
        """
        hook_handles = []

        print("\n" + "="*80)
        print("SETTING UP SPARSE INFERENCE HOOKS")
        print("="*80 + "\n")

        def create_conv_hook():
            def conv_hook(module, input):
                """Convert sparse weights to dense before convolution."""
                if module.weight.is_sparse:
                    # Must replace the entire parameter, not just .data
                    dense_weight = module.weight.to_dense()
                    module.weight = nn.Parameter(dense_weight, requires_grad=False)
                return input
            return conv_hook

        # Helper function to create Linear forward method (avoids closure issues)
        def create_sparse_linear_forward(original_forward):
            def sparse_linear_forward(self, input):
                """
                Custom forward pass for Linear layers using sparse matrix multiplication.

                Uses torch.sparse.mm() for sparse-dense matrix multiplication which can
                provide speedup for high sparsity levels (>70%).

                Math:
                    Linear layer: output = input @ weight.T + bias
                    where weight is sparse (COO format)

                Implementation:
                    torch.sparse.mm() performs: sparse_matrix @ dense_matrix
                    We compute: weight @ input.T, then transpose result
                    This is equivalent to: input @ weight.T
                """
                if self.weight.is_sparse:
                    # Use sparse matrix multiplication
                    # Linear layer: output = input @ weight.T + bias
                    # input shape: (batch_size, in_features)
                    # weight shape: (out_features, in_features) - sparse

                    # torch.sparse.mm() requires: sparse_matrix @ dense_matrix
                    # So we compute: weight @ input.T, then transpose result
                    # weight: (out_features, in_features) - sparse
                    # input.T: (in_features, batch_size) - dense
                    # Result: (out_features, batch_size), then transpose to (batch_size, out_features)

                    if input.dim() == 2:
                        # Standard case: (batch_size, in_features)
                        output = torch.sparse.mm(self.weight, input.t()).t()
                    else:
                        # Flatten input if needed
                        input_shape = input.shape
                        batch_size = input_shape[0]
                        input_flat = input.view(batch_size, -1)
                        output = torch.sparse.mm(self.weight, input_flat.t()).t()

                    # Add bias if present
                    if self.bias is not None:
                        output = output + self.bias

                    return output
                else:
                    # If not sparse, use original forward
                    return original_forward(input)
            return sparse_linear_forward

        # Register forward pre-hooks for layers with sparse weights
        for layer_name, layer_module in self.model.named_modules():
            if layer_name in sparse_tensors:
                if isinstance(layer_module, nn.Conv2d):
                    # Conv2d: Must convert sparse to dense (no sparse conv support)
                    handle = layer_module.register_forward_pre_hook(create_conv_hook())
                    hook_handles.append(handle)
                    print(f"✓ {layer_name} (Conv2d): Hook registered (sparse→dense conversion)")

                elif isinstance(layer_module, nn.Linear):
                    # Linear: Use torch.sparse.mm() for sparse matrix multiplication

                    # Store original forward method and create custom forward
                    original_forward = layer_module.forward
                    custom_forward = create_sparse_linear_forward(original_forward)

                    # Replace forward method with custom implementation
                    # Use types.MethodType to bind the function to the instance
                    import types
                    layer_module.forward = types.MethodType(custom_forward, layer_module)

                    print(f"✓ {layer_name} (Linear): Forward method replaced (torch.sparse.mm() enabled)")

        print(f"\n✓ Total hooks registered: {len(hook_handles)}")
        print("="*80 + "\n")

        return hook_handles

    def profile_sparse_model(self, sparse_tensors: dict, test_loader: DataLoader,
                            train_loader: DataLoader, device: str):
        """
        Profile model with sparse tensors to measure impact on inference.

        This method:
        1. Replaces weights with sparse tensors (stored as sparse in model)
        2. Sets up forward hooks to convert sparse→dense during inference
        3. Uses torch.sparse.mm() for Linear layers (potential speedup)
        4. Profiles memory, latency, and accuracy
        5. Compares with baseline metrics

        Parameters:
            sparse_tensors (dict): Dictionary of sparse weight tensors
            test_loader (DataLoader): Test data loader
            train_loader (DataLoader): Train data loader
            device (str): Device to run on

        Returns:
            dict: Profiling metrics
        """
        print("\n" + "="*80)
        print("PROFILING SPARSE MODEL")
        print("="*80 + "\n")

        # Step 1: Replace dense weights with sparse tensors
        self.replace_weights_with_sparse(sparse_tensors)

        # Step 2: Setup forward hooks for sparse→dense conversion during inference
        hook_handles = self.setup_sparse_inference_hooks(sparse_tensors)

        # Step 3: Profile the model
        print("Running comprehensive profiling...\n")
        self.execute_task0(train_loader, test_loader, device)

        # Step 4: Cleanup hooks
        for handle in hook_handles:
            handle.remove()

        print("\n" + "="*80)
        print("✓ Sparse model profiling complete")
        print("="*80 + "\n")
