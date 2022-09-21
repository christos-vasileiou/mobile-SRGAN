import wandb

def main():
    wandb.init(project="Testing", entity="chrivasileiou")
    wandb.save('./ax-stats-.csv')
    wandb.finish()

if __name__ == "__main__":
    main()