import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file
data = pd.read_csv('outdir/real_smoke_fire/model_real_smoke_fire_new/history.csv')

# Convert the 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])

# Plotting the data
def plot_cycle_gan_training_data(data):
    epochs = data['epoch']
    
    # Plot generator losses
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, data['gen_ab'], label='Generator AB Loss')
    plt.plot(epochs, data['gen_ba'], label='Generator BA Loss')
    plt.plot(epochs, data['cycle_a'], label='Cycle A Loss')
    plt.plot(epochs, data['cycle_b'], label='Cycle B Loss')
    plt.title('Generator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot identity losses
    plt.subplot(3, 1, 2)
    plt.plot(epochs, data['idt_a'], label='Identity A Loss')
    plt.plot(epochs, data['idt_b'], label='Identity B Loss')
    plt.title('Identity Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot discriminator losses
    plt.subplot(3, 1, 3)
    plt.plot(epochs, data['disc_a'], label='Discriminator A Loss')
    plt.plot(epochs, data['disc_b'], label='Discriminator B Loss')
    plt.title('Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Call the function to plot the data
plot_cycle_gan_training_data(data)
