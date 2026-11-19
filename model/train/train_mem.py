from model.train.train import train

if __name__ == "__main__":
    try:
        import flash_attn  # noqa: F401
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = None
    train(attn_implementation=attn_implementation)
