import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
import accelerate
import warnings
warnings.filterwarnings("ignore")

HF_HOME = os.getenv("HF_HOME", "./hf_models")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)


class DummyDataset(Dataset):
    def __init__(self, num_samples=640, num_tokens=256, max_len=256, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.input_ids = torch.randint(0, num_tokens, size=(num_samples, max_len))
        self.decoder_input_ids = torch.randint(0, num_tokens, size=(num_samples, max_len))
        self.labels = torch.randint(0, num_tokens, size=(num_samples, max_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
            "labels": self.labels[idx],
        }

def main():
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=HF_HOME).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=2,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    max_batch_size = 8
    print(f"Using batch size {max_batch_size}")
    dataloader = DataLoader(dataset, batch_size=max_batch_size, shuffle=True)
    

    
    epochs = 2
    for epoch in range(epochs):
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for data in batch_bar:
            data = {k: v.to(device) for k, v in data.items()}

            optimizer.zero_grad()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()


            with accelerator.autocast():
                output = model(
                    input_ids=data["input_ids"],
                    decoder_input_ids=data["decoder_input_ids"],
                    labels=data["labels"],  # Ensure labels are passed for loss calculation
                    return_dict=True,  # Ensure output is a dictionary
                )
                loss = output.loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            end_time.record()

            batch_tokens = max_batch_size * data["input_ids"].size(1)
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            tokens_per_second = batch_tokens / elapsed_time if elapsed_time > 0 else 0

            batch_bar.set_postfix(
                loss=loss.item(),
                tokens_per_second=f"{tokens_per_second:.2f}",
            )


if __name__ == "__main__":
    main()