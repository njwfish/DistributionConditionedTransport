"""
Print the vocabulary (token -> token ID) for the ProGen2 model.
"""

from transformers import AutoTokenizer

def main():
    progen_name = 'hugohrban/progen2-small'
    
    print(f"Loading tokenizer from: {progen_name}")
    tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    
    print(f"\nVocabulary size: {len(vocab)}")
    print("-" * 40)
    print(f"{'Token':<20} {'Token ID':>10}")
    print("-" * 40)
    
    # Sort by token ID for easier reading
    for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        # Escape special characters for display
        display_token = repr(token) if not token.isprintable() or token.isspace() else token
        print(f"{display_token:<20} {token_id:>10}")

if __name__ == "__main__":
    main()
