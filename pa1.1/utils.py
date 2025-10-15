import io
import os
import platform
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

def enablePytorchFallback():
    """
    Enable MPS fallback to CPU for unsupported operations on macOS.

    What does this do?
    PYTORCH_ENABLE_MPS_FALLBACK=1 tells PyTorch:
    "If an operation is not supported on MPS, automatically fall back to CPU"
    Note: This must be called BEFORE importing torch
    """
    # Check if running on macOS (Darwin is the system name for macOS)
    if platform.system() == 'Darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("✓ MPS fallback enabled: Unsupported operations will run on CPU")


def getDeviceFromHardware():
    
    """
    Automatically detect and return the best available device for PyTorch.

    Priority order:
    1. CUDA (NVIDIA GPUs) - Fastest, most widely supported
    2. MPS (Apple Silicon M1/M2/M3) - Fast on Mac, some ops not supported
    3. CPU - Slowest, but always available
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    import torch

    # Check CUDA availability first (highest priority)
    if torch.cuda.is_available():
        return 'cuda'

    # Check MPS availability (Apple Silicon)
    elif torch.backends.mps.is_available():
        return 'mps'

    # Fall back to CPU
    else:
        return 'cpu'


def get_base_path(gdrive_relative_path=None):
    """
    Behavior:
    - On Colab: Mounts Google Drive and returns the appropriate path
    - On Local: Returns current directory ('./')

    Parameters:
        gdrive_relative_path (str, optional): Relative path within Google Drive's MyDrive folder.

    Returns:
        str: Base path for the project
    """
    # Check if running on Google Colab
    try:
        import google.colab # type: ignore
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        # Mount Google Drive
        print("Detected Google Colab environment")
        print("Mounting Google Drive...")
        from google.colab import drive # type: ignore
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")

        # If user provided a relative path, use it
        if gdrive_relative_path:
            base_path = os.path.join('/content/drive/MyDrive', gdrive_relative_path)
            print(f"Using specified path: {base_path}")
            return base_path
        return '/content/drive/MyDrive'
    else:
        # Running locally, use current directory
        return './'


# Base Class for VGG16 on CIFAR-10 and CIFAR-100
class VGG16_CIFAR:
    """
    Helper class for managing VGG16-BN models adapted for CIFAR-10 and CIFAR-100 datasets.

    Architecture modifications from standard VGG16:
    - Modified avgpool: AdaptiveAvgPool2d((1, 1)) to output 512 features
    - Custom classifier: 512 -> 512 -> num_classes
    """

    def __init__(self, num_classes: int, is_pruned: bool = False):
        """
        Initialize VGG16_CIFAR and create the model architecture.

        Parameters:
            num_classes (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
            is_pruned (bool): Whether this is a pruned model (default: False)
        """
        self.num_classes = num_classes
        self.is_pruned = is_pruned

        # Load VGG16-BN as base
        self.model = models.vgg16_bn(num_classes=self.num_classes)

        # Modified avgpool to output 512 features (1x1 spatial dimensions)
        # Standard VGG16 uses AdaptiveAvgPool2d((7, 7)) which gives 25088 features
        # We use (1, 1) to get 512 features for CIFAR's smaller input size
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classifier for CIFAR datasets
        # Architecture: 512 -> 512 -> 512 -> num_classes
        self.model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes),
        )

    def load(self, path: str, device: str):
        """
        Load model weights from a saved checkpoint.

        Parameters:
            path (str): Path to the saved model file
            device (str): Device to load the model to ('cpu', 'cuda', or 'mps')
        """
        print(f'Loading model from {path}')

        if self.is_pruned:
            # Pruned models are saved with additional metadata
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Standard models are saved as state_dict only
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)

        # Move model to specified device
        self.model.to(device)

        # Set to evaluation mode
        self.model.eval()

        print(f'Model loaded successfully')

    def save(self, path: str):
        """
        Save model weights to a file.

        Parameters:
            path (str): Path to save the model file
        """
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print(f'Saving model to {path}')

        if self.is_pruned:
            # Save pruned models with additional metadata
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'is_pruned': self.is_pruned
            }
            torch.save(checkpoint, path)
        else:
            # Save standard models as state_dict only
            torch.save(self.model.state_dict(), path)

        print(f'Model saved successfully')

    def get_train_test_split(self, base_path: str):
        """
        Get train and test datasets for CIFAR-10 or CIFAR-100.

        Parameters:
            base_path (str): Base path for stored datasets

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        # CIFAR-10 normalization parameters from config
        # Reference: https://github.com/chenyaofo/image-classification-codebase
        # mean: [0.4914, 0.4822, 0.4465], std: [0.2023, 0.1994, 0.2010]
        if self.num_classes == 10:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            # CIFAR-100 normalization parameters from config
            # mean: [0.5070, 0.4865, 0.4409], std: [0.2673, 0.2564, 0.2761]
            mean = [0.5070, 0.4865, 0.4409]
            std = [0.2673, 0.2564, 0.2761]

        # Define transformations
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        # Dataset root directory
        root = os.path.join(base_path, 'datasets')

        # Check if dataset already exists
        # CIFAR-10 creates 'cifar-10-batches-py' directory
        # CIFAR-100 creates 'cifar-100-python' directory
        if self.num_classes == 10:
            dataset_dir = os.path.join(root, 'cifar-10-batches-py')
        else:
            dataset_dir = os.path.join(root, 'cifar-100-python')

        # Set download flag based on whether dataset exists
        download = not os.path.exists(dataset_dir)

        if download:
            print(f'Dataset not found. Downloading to {root}...')
        else:
            print(f'Dataset found at {dataset_dir}')

        # Select dataset based on num_classes
        if self.num_classes == 10:
            # CIFAR-10
            train_dataset = torchvision.datasets.CIFAR10(
                root=root,
                train=True,
                transform=train_transform,
                download=download
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=root,
                train=False,
                transform=val_transform,
                download=download
            )
        else:
            # CIFAR-100
            train_dataset = torchvision.datasets.CIFAR100(
                root=root,
                train=True,
                transform=train_transform,
                download=download
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=root,
                train=False,
                transform=val_transform,
                download=download
            )

        return train_dataset, test_dataset

    def get_data_loaders(self, train_set: Dataset, test_set: Dataset, batch_size: int = 128):
        """
        Create DataLoaders for training and testing datasets.

        Parameters:
            train_set (Dataset): Training dataset
            test_set (Dataset): Testing dataset
            batch_size (int): Batch size for both train and test loaders (default: 128)

        Returns:
            tuple: (train_loader, test_loader)
        """
        # Only use pin_memory for CUDA devices, not MPS
        use_pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        return train_loader, test_loader

    def profile_model(self, dataloader: DataLoader, device: str):
        """
        Profile the model for memory usage and inference latency.

        Parameters:
            dataloader (DataLoader): DataLoader to get input batch from
            device (str): Device to run profiling on ('cpu', 'cuda', or 'mps')

        Returns:
            tuple: (peak_memory_mb, average_memory_mb, avg_latency_ms)
        """
        # Get one batch for profiling
        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(device)

        # Measure latency outside of profiler for more accurate timing
        latencies_ms = []
        with torch.no_grad():
            for _ in range(10):  # Run 10 iterations for average
                start_time = time.time()
                output = self.model(inputs)
                end_time = time.time()
                latencies_ms.append((end_time - start_time) * 1000)

        # Calculate average latency
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)

        # Select profiler activities based on device
        if device == 'cuda':
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            memory_key = "cuda_memory_usage"
        else:
            # For 'cpu' and 'mps', use CPU profiling
            activities = [ProfilerActivity.CPU]
            memory_key = "cpu_memory_usage"

        # Profile with memory tracking
        with profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                self.model(inputs)

        # Extract memory statistics using list comprehension
        # NOTE: prof.key_averages() returns a LIST of Event objects, not a dictionary
        # Each Event represents an aggregated profiling entry for a unique operation type

        # Get all memory values in MB from operations that have memory usage
        memory_values_mb = [
            getattr(event, memory_key) / (1024 * 1024)  # Get memory and convert to MB
            for event in prof.key_averages()
            if hasattr(event, memory_key) and getattr(event, memory_key) > 0
        ]

        # Calculate metrics
        peak_memory_mb = max(memory_values_mb) if memory_values_mb else 0.0
        average_memory_mb = sum(memory_values_mb) / len(memory_values_mb) if memory_values_mb else 0.0

        return peak_memory_mb, average_memory_mb, avg_latency_ms

    def get_energy_footprint(self, dataloader: DataLoader, device: str):
        """
        Measure energy consumption during model inference.

        Note: Only works on Linux with RAPL support. Does not work on macOS.
        On systems with NVIDIA GPUs, will also measure GPU energy.

        Parameters:
            dataloader (DataLoader): DataLoader to get input batch from
            device (str): Device to run inference on ('cpu', 'cuda', or 'mps')

        Returns:
            dict: Energy measurements or None if measurement fails
        """
        try:
            from pyJoules.energy_meter import measure_energy, Domain
            from pyJoules.device.rapl_device import RaplPackageDomain
            from pyJoules.exception import NoSuchDeviceError
        except ImportError:
            print("pyJoules not installed. Install with: pip install pyJoules")
            return None

        # Setup energy measurement domains
        try:
            domains: list[Domain] = [RaplPackageDomain(0)]  # CPU Domain
        except NoSuchDeviceError:
            return None

        # Try to add GPU domain if available
        try:
            from pyJoules.device.nvidia_device import NvidiaGPUDomain
            domains.append(NvidiaGPUDomain(0))
            print("GPU energy measurement enabled")
        except Exception as e:
            pass

        # Define inference function with energy measurement decorator
        try:
            @measure_energy(domains=domains)
            def run_inference():
                # Get one batch for inference
                inputs, _ = next(iter(dataloader))
                inputs = inputs.to(device)

                with torch.no_grad():
                    for _ in range(10):  # Run 10 iterations for average
                        self.model(inputs)

            # Run energy measurement
            print("Measuring energy consumption...")
            energy_trace = run_inference()
            return energy_trace
        except NoSuchDeviceError:
            pass
            return None
        except Exception as e:
            return None

    def size_in_mb(self) -> float:
        """
        Get model size in MB by serializing it.

        Returns:
            float: Model size in megabytes
        """
        # Create in-memory buffer (fake file in RAM)
        buffer = io.BytesIO()

        # Serialize model (save all weights & biases)
        torch.save(self.model.state_dict(), buffer)

        # Get bytes written (current position = total size)
        size_bytes = buffer.tell()

        # Convert to MB (1 MB = 1024 * 1024 bytes)
        size_mb = size_bytes / (1024 * 1024)

        return size_mb

    def count_macs(self, device: str, input_size: tuple = (1, 3, 32, 32)):
        """
        Count Multiply-Accumulate operations (MACs) for the model.

        Parameters:
            device (str): Device to run on ('cpu', 'cuda', or 'mps')
            input_size (tuple): Input tensor size (default: (1, 3, 32, 32) for CIFAR)

        Returns:
            int: Number of MACs
        """
        try:
            from torchprofile import profile_macs
        except ImportError:
            print("torchprofile not installed. Install with: pip install torchprofile")
            return 0

        # Create random input tensor
        input_tensor = torch.randn(input_size).to(device)

        # Count MACs
        macs = profile_macs(self.model, input_tensor)

        return macs

    def profile_accuracy(self, data_loader: DataLoader, device: str, show_progress: bool = False):
        """
        Evaluate model accuracy with Top-1 and Top-5 metrics.

        Note: This method evaluates on ALL batches in the dataloader for accurate metrics.

        Parameters:
            data_loader (DataLoader): DataLoader for evaluation
            device (str): Device to run evaluation on ('cpu', 'cuda', or 'mps')
            show_progress (bool): Whether to show progress bar (default: False)

        Returns:
            tuple: (top1_accuracy, top5_accuracy) in percentages
        """
        self.model.eval()

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            # Loop through ALL batches in the dataloader
            iterator = tqdm(data_loader, desc="Validating", unit="batch", leave=False) if show_progress else data_loader

            for images, labels in iterator:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)

                # Top-1 accuracy
                _, predicted_top1 = torch.max(outputs, 1)
                correct_top1 += (predicted_top1 == labels).sum().item()

                # Top-5 accuracy
                _, predicted_top5 = torch.topk(outputs, 5, dim=1)
                correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(len(labels))])

                total += labels.size(0)

        top1_accuracy = 100 * correct_top1 / total
        top5_accuracy = 100 * correct_top5 / total

        return top1_accuracy, top5_accuracy

    def execute_task0(self, train_loader: DataLoader, test_loader: DataLoader, device: str):
        """
        Execute Task 0: Comprehensive baseline model profiling.

        Runs all profiling methods and prints formatted results including:
        - Model size
        - Memory usage (peak and average)
        - Inference latency
        - Energy consumption (if available)
        - MACs
        - Accuracy metrics (Top-1 and Top-5 for both train and test)

        Parameters:
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            device (str): Device to run profiling on ('cpu', 'cuda', or 'mps')
        """
        print("\n" + "="*80)
        print("TASK 0: BASELINE MODEL PROFILING RESULTS")
        print("="*80)

        dataset_name = "CIFAR-10" if self.num_classes == 10 else "CIFAR-100"
        print(f"\n--- {dataset_name} VGG16-BN ---")

        # Model size
        model_size_mb = self.size_in_mb()
        print(f"Model Size:          {model_size_mb:.2f} MB")

        # Memory and latency profiling
        peak_mem, avg_mem, latency = self.profile_model(test_loader, device)
        print(f"Peak Memory:         {peak_mem:.2f} MB")
        print(f"Average Memory:      {avg_mem:.2f} MB")
        print(f"Latency (per batch): {latency:.2f} ms")

        # Energy profiling (may not work on all systems)
        energy_trace = self.get_energy_footprint(test_loader, device)
        if energy_trace:
            print(f"Energy:              {energy_trace}")
        else:
            print(f"Energy:              Not available (Linux with RAPL required)")

        # MACs counting
        macs = self.count_macs(device)
        print(f"MACs:                {macs:,}")

        # Accuracy profiling
        print("\nEvaluating test accuracy...")
        test_top1, test_top5 = self.profile_accuracy(test_loader, device)
        print(f"Test Top-1:          {test_top1:.2f}%")
        print(f"Test Top-5:          {test_top5:.2f}%")

        print("\nEvaluating train accuracy...")
        train_top1, train_top5 = self.profile_accuracy(train_loader, device)
        print(f"Train Top-1:         {train_top1:.2f}%")
        print(f"Train Top-5:         {train_top5:.2f}%")

        print("\n" + "="*80)

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        optimizer,
        scheduler,
        criterion,
        target_accuracy: float,
        checkpoint_path: str,
        max_epochs: int = 200,
        early_stop_patience: int = 15,
        pre_step_callback = None, # type: ignore
        post_step_callback = None # type: ignore
    ) -> float:
        """
        Fine-tune model with provided optimizer, scheduler, and criterion.

        This method provides:
        - Pure training loop with no hardcoded defaults
        - Checkpointing and resume capability
        - Early stopping to prevent overfitting
        - Target accuracy stopping condition
        - Gradient clipping for stability (max_norm=5.0)
        - Optional pre/post-step callbacks for custom logic

        Training loop execution order (per batch):
            1. Forward pass: outputs = model(inputs)
            2. Loss computation: loss = criterion(outputs, labels)
            3. Backward pass: loss.backward()
            4. Gradient clipping: clip_grad_norm_()
            5. **pre_step_callback(model)** ← Modify gradients here
            6. Weight update: optimizer.step()
            7. **post_step_callback(model)** ← Modify weights here

        Parameters:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            device (str): Device to train on ('cpu', 'cuda', 'mps')
            optimizer (torch.optim.Optimizer): Optimizer instance (required)
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler instance (required)
            criterion (nn.Module): Loss function instance (required)
            target_accuracy (float): Stop training when this accuracy is reached
            checkpoint_path (str): Path to save/load checkpoints
            max_epochs (int): Maximum number of training epochs (default: 200)
            early_stop_patience (int): Stop if no improvement for N epochs (default: 15)
            pre_step_callback (callable, optional): Function called before optimizer.step()
                Signature: callback(model) - useful for gradient manipulation
            post_step_callback (callable, optional): Function called after optimizer.step()
                Signature: callback(model) - useful for weight modifications (e.g., sparsity)

        Returns:
            float: Best validation Top-1 accuracy achieved
        """
        print("="*80)
        print(f"Fine-tuning to {target_accuracy}% accuracy (max {max_epochs} epochs)")
        print("="*80)

        # Move model to device
        self.model.to(device)
        self.model.train()

        # Load checkpoint if exists
        start_epoch = 0
        best_accuracy = 0.0

        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            print(f"✓ Resumed from epoch {start_epoch}, best accuracy: {best_accuracy:.2f}%")

            # Check if target accuracy already achieved
            if best_accuracy >= target_accuracy:
                print(f"\n✓ Target accuracy {target_accuracy}% already achieved!")
                print(f"✓ Checkpoint has accuracy: {best_accuracy:.2f}%")
                print(f"✓ Skipping training - returning existing model")
                print("="*80)
                return best_accuracy

        # Early stopping tracking
        epochs_without_improvement = 0

        print("\nStarting training...\n")

        for epoch in range(start_epoch, max_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0

            # Create progress bar for batches
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}",
                               unit="batch", leave=False)

            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # Call pre-step callback if provided (e.g., for gradient manipulation)
                if pre_step_callback is not None:
                    pre_step_callback(self.model)

                optimizer.step()

                # Call post-step callback if provided (e.g., for sparsity enforcement)
                if post_step_callback is not None:
                    post_step_callback(self.model)

                running_loss += loss.item()

                # Update progress bar with current loss
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Step the learning rate scheduler
            scheduler.step()

            # Validation phase
            val_top1, val_top5 = self.profile_accuracy(val_loader, device, show_progress=True)
            avg_loss = running_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch [{epoch+1}/{max_epochs}] Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%")

            # Check if target accuracy reached
            if val_top1 >= target_accuracy:
                print(f"\n✓ Target accuracy {target_accuracy}% reached! (Val Top-1: {val_top1:.2f}%)")
                best_accuracy = val_top1

                # Save final checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy
                }, checkpoint_path)
                print(f"✓ Final checkpoint saved to {checkpoint_path}")

                break

            # Track best accuracy and early stopping
            if val_top1 > best_accuracy:
                best_accuracy = val_top1
                epochs_without_improvement = 0

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping: No improvement for {early_stop_patience} epochs")
                print(f"Best accuracy: {best_accuracy:.2f}%")
                break

        print("="*80)
        print(f"Fine-tuning completed. Best accuracy: {best_accuracy:.2f}%")
        print("="*80)

        return best_accuracy