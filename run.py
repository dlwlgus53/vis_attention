import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification

import numpy as np
import matplotlib.pyplot as plt


def visualize_matrix(matrix, axis, save_path):
    n = matrix.shape[0]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the matrix as an image with a colormap
    cax = ax.matshow(matrix, cmap="Reds")

    # Add a colorbar to show the mapping of values to colors
    cbar = fig.colorbar(cax)

    # Set labels for x and y axes
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(axis, rotation=90)
    ax.set_yticklabels(axis)

    # Display the values in each cell of the matrix

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# # Example usage:
# # Replace 'your_matrix' with your actual NumPy matrix
# your_matrix = np.random.rand(5, 5) * 10  # Replace this line with your actual matrix
# save_path = "matrix_visualization.png"
# visualize_matrix(your_matrix, save_path)

if __name__ == "__main__":
    text = ["Look at the bridge, it is falling down"]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER",
        output_hidden_states=True,
        output_attentions=True,
    )

    with torch.no_grad():
        batch = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        token_length = len(batch["input_ids"][0])

        attention_matrix = np.zeros([token_length, token_length])

        token_list = [
            tokenizer.decode(batch["input_ids"][0][i]) for i in range(token_length)
        ]

        for layer in range(8, 12):  # 12 layers
            attention = outputs["attentions"][layer][0].detach().numpy()
            attention = np.sum(attention, axis=0)
            attention_matrix += attention

        # remove first and last token
        attention_matrix = attention_matrix[1:-1, 1:-1]
        token_list = token_list[1:-1]
        visualize_matrix(attention_matrix, token_list, "attention_matrix.png")
