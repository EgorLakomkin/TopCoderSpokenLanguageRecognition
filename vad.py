def remove_silence(samples, threshold):
    samples_abs = [abs(sample) for sample in samples]

    # moving average for samples
    average = []
    alpha = 0.001
    prev = 0
    for sample in samples_abs:
        y = alpha * sample + (1 - alpha) * prev
        prev = y
        average.append(y)

    voice_samples = [s for i, s in enumerate(samples) if average[i] > threshold]

    return voice_samples
