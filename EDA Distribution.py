import os
import matplotlib.pyplot as plt


def plot_emotion_distribution(base_path, output_graph_path):
    # Define emotions
    ecgs = ['F', 'N', 'Q', 'S', 'V']

    # Initialize a dictionary to hold the count of images for each emotion
    ecg_counts = {ecg: 0 for ecg in ecgs}

    # Count the number of images in each emotion's directory
    for ecg in ecgs:
        ecg_dir = os.path.join(base_path, ecg)
        try:
            ecg_counts[ecg] = len(
                [name for name in os.listdir(ecg_dir) if os.path.isfile(os.path.join(ecg_dir, name))])
        except FileNotFoundError:
            print(f"Directory not found for {ecg}, setting count to 0.")
            ecg_counts[ecg] = 0

    # Plotting the distribution of images across emotions
    plt.figure(figsize=(10, 6))
    plt.bar(ecg_counts.keys(), ecg_counts.values(), color='skyblue')
    plt.xlabel('ECG Signal')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images by ECG signal')
    plt.xticks(rotation=45)

    # Save the plot
    plt.savefig(output_graph_path)
    plt.close()


# Example usage
base_path = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/ECG_Image_data/data'
output_path = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/ECG_Distribution.png'
plot_emotion_distribution(base_path, output_path)