import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional

# Import VGG16_Pruning to inherit COO sparse methods
from task1a import VGG16_Pruning


class VGG16_GraSP(VGG16_Pruning):
    """
    Extends VGG16_Pruning with GraSP saliency-based iterative pruning.

    Inherits from task1a (VGG16_Pruning):
    - convert_to_coo_sparse()
    - verify_mask_coo_consistency()
    - setup_sparse_inference_hooks()
    - profile_sparse_model()
    - All other COO-related methods

    Adds GraSP-specific methods:
    - Random weight initialization
    - Hessian-gradient product computation (Algorithm 2 from GraSP paper)
    - GraSP pruning with score computation (Algorithm 1 from GraSP paper)
    - 3-stage iterative pruning pipeline
    """

    def __init__(self, num_classes: int, is_pruned: bool = False):
        """
        Initialize VGG16_GraSP with parent class initialization.

        Parameters:
            num_classes (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
            is_pruned (bool): Whether this is a pruned model (default: False)
        """
        # Call parent constructor to set up VGG16 architecture and inherit COO methods
        super().__init__(num_classes, is_pruned)

    def initialize_random_weights(self):
        """
        Reset model to random weights using Kaiming initialization.

        This method reinitializes all Conv2d, Linear, and BatchNorm layers to random weights,
        simulating training from scratch.

        Initialization schemes:
        - Conv2d and Linear: Kaiming normal (He initialization) with fan_out mode
        - BatchNorm2d: Weight=1, Bias=0 (standard practice)

        Reference: https://pytorch.org/docs/stable/nn.init.html
        """
        print("Initializing model with random weights...")

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming normal for convolutional layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                # Kaiming normal for linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for batch normalization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        print("✓ Model weights initialized with random values")

    def compute_hessian_gradient_product(self, calibration_loader: DataLoader,
                                        device: str) -> Dict[str, torch.Tensor]:
        """
        Implement Algorithm 2 from GraSP paper.

        Computes the Hessian-gradient product: Hg = ∇(g^T · stop_grad(g))
        where g = ∇L(θ₀) is the gradient of the loss w.r.t. initial weights.

        This captures how weights interact (via Hessian) and identifies which weights
        are critical for gradient flow preservation.

        Mathematical Formulation:
        1. Compute loss: L(θ₀)
        2. Compute first-order gradient: g = ∇L(θ₀)
        3. Compute scalar product: s = g^T · stop_grad(g)
        4. Compute second-order gradient: Hg = ∇s

        The result Hg tells us how each weight affects the gradient flow.

        Parameters:
            calibration_loader (DataLoader): DataLoader with calibration samples
            device (str): Device to run computation

        Returns:
            dict: Dictionary mapping parameter names to Hg tensors
        """
        print("Computing Hessian-gradient product (Algorithm 2)...")
        print(f"  Calibration batches: {len(calibration_loader)}")

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        # Initialize accumulated Hessian-gradient product
        accumulated_hg = None
        num_batches = 0

        # Process calibration batches
        for images, labels in calibration_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Step 1: Forward pass and compute loss L(θ₀)
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Step 2: Compute gradient g = ∇L(θ₀)
            # create_graph=True keeps computation graph for second derivative
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=self.model.parameters(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )

            # Step 3: Compute g^T · stop_grad(g)
            # stop_grad(g) = g.detach() in PyTorch
            # g^T · stop_grad(g) is a scalar sum of element-wise products
            gradient_product = sum([
                torch.sum(g * g.detach())
                for g in gradients
            ])

            # Step 4: Compute Hg = ∇(g^T · stop_grad(g))
            # This is the second derivative (Hessian-gradient product)
            # NOTE: Don't pass retain_graph parameter, let PyTorch handle it
            hessian_gradient = torch.autograd.grad(
                outputs=gradient_product,
                inputs=self.model.parameters(),
                only_inputs=True
            )

            # Accumulate Hg across batches (use list like backup)
            if accumulated_hg is None:
                # First batch - initialize as list
                accumulated_hg = [hg.detach().clone() for hg in hessian_gradient]
            else:
                # Subsequent batches - accumulate
                for i, hg in enumerate(hessian_gradient):
                    accumulated_hg[i] += hg.detach()

            num_batches += 1

        accumulated_hg = [hg / num_batches for hg in accumulated_hg]

        # Convert to dictionary mapping layer names to Hg tensors
        hessian_gradient_product = {}
        for (name, param), hg in zip(self.model.named_parameters(), accumulated_hg):
            if 'weight' in name:  # Only track weights, not biases
                hessian_gradient_product[name] = hg

        print(f"✓ Hessian-gradient product computed over {num_batches} batches")
        print(f"  Layers processed: {len(hessian_gradient_product)}")

        return hessian_gradient_product

    def grasp_prune(self, calibration_loader: DataLoader, device: str,
                   pruning_ratio: float, existing_mask: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Implement Algorithm 1 from GraSP paper.

        Prunes weights based on gradient signal preservation criterion:
            S(-θ) = -θ ⊙ Hg

        Removes weights with highest scores (those whose removal minimally impacts gradient flow).

        Score Interpretation:
        - Lower score = more important for gradient flow → keep
        - Higher score = less important → prune

        Process:
        1. Compute Hessian-gradient product using Algorithm 2
        2. Compute scores: S(-θ) = -θ ⊙ Hg (element-wise multiplication)
        3. Find threshold at pruning_ratio percentile
        4. Create binary mask: keep if score < threshold, prune otherwise
        5. Combine with existing_mask if provided (frozen mask from previous stage)

        Parameters:
            calibration_loader (DataLoader): DataLoader with calibration samples
            device (str): Device to run on ('cpu', 'cuda', or 'mps')
            pruning_ratio (float): Fraction of weights to prune (0.0 to 1.0)
            existing_mask (dict, optional): Mask from previous pruning stage (frozen)

        Returns:
            dict: Dictionary mapping layer names to binary masks (1=keep, 0=prune)
        """
        print("="*80)
        print(f"GraSP Pruning (Algorithm 1) - Target sparsity: {pruning_ratio*100}%")
        print("="*80)

        # Step 1 & 2: Compute Hessian-gradient product using Algorithm 2
        hessian_gradient_product = self.compute_hessian_gradient_product(calibration_loader, device)

        # Step 3: Compute scores S(-θ) = -θ ⊙ Hg
        # Collect scores ONLY for weights that can still be pruned
        all_scores = []
        layer_scores = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name and name in hessian_gradient_product:
                # Get parameter name without '.weight' suffix to match layer names
                layer_name = name.replace('.weight', '')

                # Compute scores: S(-θ) = -θ ⊙ Hg (element-wise multiplication)
                scores = -param.data * hessian_gradient_product[name]

                layer_scores[layer_name] = scores

                # CRITICAL: Only include scores for weights that can still be pruned
                # If existing_mask provided, only collect scores where mask == 1 (unpruned)
                if existing_mask is not None and layer_name in existing_mask:
                    # Extract scores only for unpruned weights (mask == 1)
                    unpruned_scores = scores[existing_mask[layer_name] == 1]
                    all_scores.append(unpruned_scores.view(-1))
                else:
                    # No existing mask - all weights can be pruned
                    all_scores.append(scores.view(-1))

        # Concatenate all scores into single tensor for global thresholding
        all_scores_tensor = torch.cat(all_scores)

        # Step 4: Find threshold τ at p-th percentile
        # Calculate based on number of PRUNABLE weights, not total weights
        # pruning_ratio=0.8 means we want to keep 20% of PRUNABLE weights
        num_params_to_keep = int((1 - pruning_ratio) * all_scores_tensor.numel())

        # Move to CPU for kthvalue if on MPS device (not supported)
        if device == 'mps':
            all_scores_cpu = all_scores_tensor.cpu()
            threshold = torch.kthvalue(all_scores_cpu, num_params_to_keep).values.item()
        else:
            threshold = torch.kthvalue(all_scores_tensor, num_params_to_keep).values.item()

        print(f"\nComputed threshold: {threshold:.6f}")
        print(f"Prunable parameters: {all_scores_tensor.numel():,}")
        print(f"Parameters to keep: {num_params_to_keep:,} ({(1-pruning_ratio)*100:.1f}% of prunable)")
        print(f"Parameters to prune: {all_scores_tensor.numel() - num_params_to_keep:,} ({pruning_ratio*100:.1f}% of prunable)")

        # Step 5: Create masks m = (S(-θ) < τ)
        # Keep weights with scores BELOW threshold (lower scores = more important)
        pruning_mask = {}

        for layer_name, scores in layer_scores.items():
            # Create binary mask: 1 where score < threshold (keep), 0 otherwise (prune)
            mask = (scores < threshold).float()

            # Combine with existing mask if provided (frozen mask from previous stage)
            if existing_mask is not None and layer_name in existing_mask:
                mask = mask * existing_mask[layer_name]

            pruning_mask[layer_name] = mask

            # Report sparsity for this layer
            num_pruned = torch.sum(mask == 0).item()
            num_total = mask.numel()
            layer_sparsity = num_pruned / num_total

            print(f"  {layer_name}: {layer_sparsity*100:.2f}% sparsity ({num_pruned:,}/{num_total:,} pruned)")

        # Calculate overall sparsity
        total_params = sum(mask.numel() for mask in pruning_mask.values())
        total_pruned = sum(torch.sum(mask == 0).item() for mask in pruning_mask.values())
        overall_sparsity = total_pruned / total_params

        print(f"\n✓ Overall sparsity: {overall_sparsity*100:.2f}% ({total_pruned:,}/{total_params:,} weights pruned)")
        print("="*80)

        return pruning_mask

    def apply_pruning_mask(self, pruning_mask: Dict[str, torch.Tensor]):
        """
        Apply pruning mask to model by setting pruned weights to zero.

        Parameters:
            pruning_mask (dict): Dictionary mapping layer names to binary masks
        """
        with torch.no_grad():
            for layer_name, layer_module in self.model.named_modules():
                if layer_name in pruning_mask:
                    if isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                        layer_module.weight.data.mul_(pruning_mask[layer_name])

    def create_mask_enforcement_callback(self, pruning_mask: Dict[str, torch.Tensor], device: str):
        """
        Create a callback function that enforces sparsity after each optimizer step.

        This callback re-applies pruning masks to maintain sparsity during training.
        Without this, the optimizer would update pruned weights back to non-zero values.

        Parameters:
            pruning_mask (dict): Dictionary of pruning masks {layer_name: mask_tensor}
            device (str): Device to run on ('cpu', 'cuda', 'mps')

        Returns:
            callable: Callback function with signature callback(model)
        """
        def mask_enforcement_callback(model: nn.Module):
            """Re-apply pruning masks to maintain sparsity after optimizer step."""
            with torch.no_grad():
                for layer_name, layer_module in model.named_modules():
                    if layer_name in pruning_mask:
                        if isinstance(layer_module, (nn.Conv2d, nn.Linear)):
                            # Ensure mask is on the same device
                            mask = pruning_mask[layer_name].to(device)
                            layer_module.weight.data.mul_(mask)

        return mask_enforcement_callback

    def _load_checkpoint_if_exists(self, checkpoint_path: str, device: str) -> Optional[float]:
        """
        Load checkpoint if it exists and return accuracy.

        Parameters:
            checkpoint_path (str): Path to checkpoint file
            device (str): Device to load checkpoint to

        Returns:
            float or None: Best accuracy from checkpoint, or None if doesn't exist
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['best_accuracy']
        return None

    def _load_mask_if_exists(self, mask_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load pruning mask if it exists.

        Parameters:
            mask_path (str): Path to mask file

        Returns:
            dict or None: Pruning mask dict, or None if doesn't exist
        """
        if os.path.exists(mask_path):
            mask = torch.load(mask_path, map_location='cpu', weights_only=True)
            self.apply_pruning_mask(mask)
            return mask
        return None

    def _run_training_stage(self, stage_name: str, target_accuracy: float, checkpoint_path: str,
                           train_loader: DataLoader, val_loader: DataLoader, device: str,
                           criterion: nn.Module, current_mask: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """
        Run a training stage with automatic checkpoint recovery.

        Parameters:
            stage_name (str): Name of stage for logging (e.g., "Stage 1", "Final")
            target_accuracy (float): Target accuracy to reach
            checkpoint_path (str): Path to save/load checkpoint
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            device (str): Device to train on
            criterion (nn.Module): Loss function
            current_mask (dict, optional): Mask to enforce during training

        Returns:
            float: Best accuracy achieved
        """
        # Try to load existing checkpoint
        best_acc = self._load_checkpoint_if_exists(checkpoint_path, device)

        if best_acc is not None:
            print(f"\n{'='*80}")
            print(f"{stage_name} TRAINING: ✓ SKIPPING (checkpoint found)")
            print(f"{'='*80}")
            print(f"✓ Loaded with accuracy: {best_acc:.2f}%\n")
            return best_acc

        # Run training
        print(f"\n{'='*80}")
        print(f"{stage_name} TRAINING: Target accuracy {target_accuracy}%")
        print(f"{'='*80}\n")

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

        # Create mask callback if mask provided
        post_callback = self.create_mask_enforcement_callback(current_mask, device) if current_mask else None

        best_acc = self.fine_tune(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            target_accuracy=target_accuracy,
            checkpoint_path=checkpoint_path,
            max_epochs=200,
            early_stop_patience=15,
            post_step_callback=post_callback
        )

        return best_acc

    def _run_pruning_stage(self, stage_name: str, pruning_ratio: float, mask_path: str,
                          calibration_loader: DataLoader, device: str,
                          existing_mask: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Run a pruning stage with automatic mask recovery.

        Parameters:
            stage_name (str): Name of stage for logging (e.g., "Stage 1", "Stage 2")
            pruning_ratio (float): Target pruning ratio
            mask_path (str): Path to save/load mask
            calibration_loader (DataLoader): Calibration data loader
            device (str): Device to run on
            existing_mask (dict, optional): Existing mask from previous stage

        Returns:
            dict: Pruning mask
        """
        # Try to load existing mask
        mask = self._load_mask_if_exists(mask_path)

        if mask is not None:
            sparsity = sum(torch.sum(m == 0).item() for m in mask.values()) / sum(m.numel() for m in mask.values())
            print(f"\n{'='*80}")
            print(f"{stage_name} PRUNING: ✓ SKIPPING (mask found)")
            print(f"{'='*80}")
            print(f"✓ Loaded with {sparsity*100:.2f}% sparsity\n")
            return mask

        # Run GraSP pruning
        print(f"\n{'='*80}")
        print(f"{stage_name} PRUNING: Target sparsity {pruning_ratio*100}%")
        print(f"{'='*80}\n")

        mask = self.grasp_prune(
            calibration_loader=calibration_loader,
            device=device,
            pruning_ratio=pruning_ratio,
            existing_mask=existing_mask
        )

        self.apply_pruning_mask(mask)

        # Save mask immediately
        torch.save(mask, mask_path)
        print(f"✓ Mask saved to {mask_path}\n")

        return mask

    def iterative_pruning_pipeline(self, train_loader: DataLoader, val_loader: DataLoader,
                                  device: str, target_sparsity: float, checkpoint_dir: str,
                                  masks_dir: str, results_dir: str, dataset_name: str) -> tuple:
        """
        Main pipeline orchestrating 3-stage iterative pruning with GraSP.

        Pipeline Flow:
        1. Initialize random weights
        2. Stage 1: Train to 20% acc → GraSP prune to 50% of target 
        3. Stage 2: Train to 40% acc → GraSP prune to 75% of target 
        4. Stage 3: Train to 60% acc → GraSP prune to 100% of target
        5. Final fine-tuning with all masks frozen

        Parameters:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            device (str): Device to run on ('cpu', 'cuda', 'mps')
            target_sparsity (float): Target overall sparsity (e.g., 0.80 for 80%)
            checkpoint_dir (str): Directory to save training checkpoints (task1b/checkpoints/)
            masks_dir (str): Directory to save pruning masks (task1b/pruning_masks/)
            results_dir (str): Directory to save JSON results (task1b/results/)
            dataset_name (str): Name of dataset ('cifar10' or 'cifar100')

        Returns:
            tuple: (final_mask, best_accuracy)
        """
        print("\n" + "="*80)
        print("TASK 1B: GRASP ITERATIVE PRUNING PIPELINE (ROBUST MODE)")
        print("="*80)
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Target Sparsity: {target_sparsity*100}%")
        print(f"Stage 1 (at 20% acc): {target_sparsity*0.50*100}% sparsity (50% of target)")
        print(f"Stage 2 (at 40% acc): {target_sparsity*0.75*100}% sparsity (75% of target)")
        print(f"Stage 3 (at 60% acc): {target_sparsity*1.00*100}% sparsity (100% of target)")
        print("="*80 + "\n")

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Define file paths
        stage1_checkpoint = os.path.join(checkpoint_dir, f"{dataset_name}_stage1_trained.pt")
        stage1_mask_path = os.path.join(masks_dir, f"{dataset_name}_mask_stage1.pt")
        stage2_checkpoint = os.path.join(checkpoint_dir, f"{dataset_name}_stage2_trained.pt")
        stage2_mask_path = os.path.join(masks_dir, f"{dataset_name}_mask_stage2.pt")
        stage3_checkpoint = os.path.join(checkpoint_dir, f"{dataset_name}_stage3_trained.pt")
        stage3_mask_path = os.path.join(masks_dir, f"{dataset_name}_mask_stage3_FINAL.pt")
        final_checkpoint = os.path.join(checkpoint_dir, f"{dataset_name}_final_finetuned.pt")
        results_path = os.path.join(results_dir, f"{dataset_name}_grasp_results.json")

        print("="*80)
        print("CHECKING FOR EXISTING CHECKPOINTS (RECOVERY MODE)")
        print("="*80)

        # Check what exists
        results_exists = os.path.exists(results_path)
        final_ckpt_exists = os.path.exists(final_checkpoint)
        stage3_mask_exists = os.path.exists(stage3_mask_path)
        stage3_ckpt_exists = os.path.exists(stage3_checkpoint)
        stage2_mask_exists = os.path.exists(stage2_mask_path)
        stage2_ckpt_exists = os.path.exists(stage2_checkpoint)
        stage1_mask_exists = os.path.exists(stage1_mask_path)
        stage1_ckpt_exists = os.path.exists(stage1_checkpoint)

        print(f"  Results JSON:         {'✓ EXISTS' if results_exists else '✗ Not found'}")
        print(f"  Final checkpoint:     {'✓ EXISTS' if final_ckpt_exists else '✗ Not found'}")
        print(f"  Stage 3 mask:         {'✓ EXISTS' if stage3_mask_exists else '✗ Not found'}")
        print(f"  Stage 3 checkpoint:   {'✓ EXISTS' if stage3_ckpt_exists else '✗ Not found'}")
        print(f"  Stage 2 mask:         {'✓ EXISTS' if stage2_mask_exists else '✗ Not found'}")
        print(f"  Stage 2 checkpoint:   {'✓ EXISTS' if stage2_ckpt_exists else '✗ Not found'}")
        print(f"  Stage 1 mask:         {'✓ EXISTS' if stage1_mask_exists else '✗ Not found'}")
        print(f"  Stage 1 checkpoint:   {'✓ EXISTS' if stage1_ckpt_exists else '✗ Not found'}")
        print("="*80 + "\n")

        # Determine resume point
        if results_exists:
            print("✓ PIPELINE ALREADY COMPLETE!")
            print(f"✓ Loading results from {results_path}\n")
            with open(results_path, 'r') as f:
                results = json.load(f)

            # Load final mask
            final_mask = torch.load(stage3_mask_path, map_location='cpu', weights_only=True)

            # Load final model checkpoint
            checkpoint = torch.load(final_checkpoint, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)

            print(f"Final Accuracy: {results['final_metrics']['top1_accuracy']:.2f}%")
            print(f"Final Sparsity: {results['final_sparsity']*100:.2f}%")
            print("="*80 + "\n")

            return final_mask, results['final_metrics']['top1_accuracy']

        # Track stage results (load if exists)
        stage_results = []
        current_mask = None

        # Create calibration loader (512 samples = 4 batches of 128)
        # Take subset of training data for calibration
        calibration_size = 512
        calibration_indices = list(range(calibration_size))
        calibration_dataset = Subset(train_loader.dataset, calibration_indices)
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )

        # Initialize random weights (only if starting fresh)
        if not (stage1_ckpt_exists or stage1_mask_exists):
            self.initialize_random_weights()
        
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Track stage results
        stage_results = []

        # Training
        best_acc_stage1 = self._run_training_stage(
            stage_name="STAGE 1",
            target_accuracy=20.0,
            checkpoint_path=stage1_checkpoint,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
        )

        # Pruning
        stage1_mask = self._run_pruning_stage(
            stage_name="STAGE 1",
            pruning_ratio=target_sparsity * 0.50,
            mask_path=stage1_mask_path,
            calibration_loader=calibration_loader,
            device=device,
        )

        # Record results
        stage_results.append({
            "stage": 1,
            "target_accuracy": 20.0,
            "achieved_accuracy": best_acc_stage1,
            "target_sparsity": target_sparsity * 0.50,
            "actual_sparsity": sum(torch.sum(m == 0).item() for m in stage1_mask.values()) / sum(m.numel() for m in stage1_mask.values())
        })

        current_mask = stage1_mask

        # Training
        best_acc_stage2 = self._run_training_stage(
            stage_name="STAGE 2",
            target_accuracy=40.0,
            checkpoint_path=stage2_checkpoint,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            current_mask=current_mask
        )

        # Pruning
        stage2_mask = self._run_pruning_stage(
            stage_name="STAGE 2",
            pruning_ratio=target_sparsity * 0.75,
            mask_path=stage2_mask_path,
            calibration_loader=calibration_loader,
            device=device,
            existing_mask=current_mask
        )

        # Record results
        stage_results.append({
            "stage": 2,
            "target_accuracy": 40.0,
            "achieved_accuracy": best_acc_stage2,
            "target_sparsity": target_sparsity * 0.75,
            "actual_sparsity": sum(torch.sum(m == 0).item() for m in stage2_mask.values()) / sum(m.numel() for m in stage2_mask.values())
        })

        current_mask = stage2_mask

        # Training
        best_acc_stage3 = self._run_training_stage(
            stage_name="STAGE 3",
            target_accuracy=60.0,
            checkpoint_path=stage3_checkpoint,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            current_mask=current_mask
        )

        # Pruning
        stage3_mask = self._run_pruning_stage(
            stage_name="STAGE 3",
            pruning_ratio=target_sparsity,
            mask_path=stage3_mask_path,
            calibration_loader=calibration_loader,
            device=device,
            existing_mask=current_mask
        )

        # Record results
        stage_results.append({
            "stage": 3,
            "target_accuracy": 60.0,
            "achieved_accuracy": best_acc_stage3,
            "target_sparsity": target_sparsity,
            "actual_sparsity": sum(torch.sum(m == 0).item() for m in stage3_mask.values()) / sum(m.numel() for m in stage3_mask.values())
        })

        current_mask = stage3_mask


        best_acc_final = self._run_training_stage(
            stage_name="FINAL",
            target_accuracy=100.0,  # High target to train full epochs
            checkpoint_path=final_checkpoint,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            current_mask=current_mask
        )

        # Evaluate final metrics
        print("\n" + "="*80)
        print("EVALUATING FINAL MODEL")
        print("="*80 + "\n")

        final_top1, final_top5 = self.profile_accuracy(val_loader, device, show_progress=True)

        # Calculate final sparsity
        final_sparsity = sum(torch.sum(m == 0).item() for m in current_mask.values()) / sum(m.numel() for m in current_mask.values())

        # Save results JSON
        results = {
            "dataset": dataset_name,
            "method": "GraSP Saliency-Based Iterative Pruning",
            "target_sparsity": target_sparsity,
            "final_sparsity": final_sparsity,
            "stages": stage_results,
            "final_metrics": {
                "top1_accuracy": final_top1,
                "top5_accuracy": final_top5,
                "model_size_mb": self.size_in_mb(),
                "sparsity_ratio": final_sparsity
            }
        }

        results_path = os.path.join(results_dir, f"{dataset_name}_grasp_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")

        print("\n" + "="*80)
        print("TASK 1B PIPELINE COMPLETE")
        print("="*80)
        print(f"Final Top-1 Accuracy: {final_top1:.2f}%")
        print(f"Final Top-5 Accuracy: {final_top5:.2f}%")
        print(f"Final Sparsity: {final_sparsity*100:.2f}%")
        print(f"Model Size: {self.size_in_mb():.2f} MB")
        print("="*80 + "\n")

        return current_mask, best_acc_final
