import os
import re
import math

video_dir = r""  

video_dir = os.path.normpath(video_dir)

# Match files in the id.mp4 format.
pattern = re.compile(r'^[a-zA-Z0-9_]+\.mp4$', re.IGNORECASE)

# Get the full path of all files that meet the criteria.
video_files = [
    os.path.join(video_dir, f)
    for f in os.listdir(video_dir)
    if pattern.match(f)
]

batch_size = 50
total_batches = math.ceil(len(video_files) / batch_size)

# Output all commands
for i in range(total_batches):
    batch = video_files[i * batch_size:(i + 1) * batch_size]
    file_args = ' '.join(f'-f "{f}"' for f in batch)
    command = f'.\\FeatureExtraction.exe {file_args} -aus'

    print(f"\n✅ Batch {i + 1}/{total_batches} Command:\n")
    print(command)
