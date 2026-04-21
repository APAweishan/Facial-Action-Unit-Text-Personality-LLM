import torch
import torch.nn as nn

class LLMWithRegressionHead(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.backbone = base_model
        self.tokenizer = tokenizer

        # Freeze non-LoRA parameters
        for name, param in self.backbone.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]  # [B, seq_len, hidden]
    
        batch_size, seq_len, hidden_size = last_hidden_state.size()
        seq_lengths = attention_mask.sum(dim=1)          # [B]
        last_token_indices = seq_lengths - 1            # [B]
        regression_input = last_hidden_state[torch.arange(batch_size), last_token_indices]

        score = self.regression_head(regression_input.to(self.regression_head[0].weight.dtype))
        return score.squeeze(-1)

