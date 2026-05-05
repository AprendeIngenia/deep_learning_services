import pandas as pd
import os
import io

# config
ARCHIVO_PARQUET = 'src/training/collect_data/files/train-00000-of-00002.parquet'

# images
image_column = 'image'

# output
output_folder = 'src/training/collect_data/images'

# format
output_format = 'jpg'

# create folder
os.makedirs(output_folder, exist_ok=True)
print(f"Cargando el archivo {ARCHIVO_PARQUET}...")

# read column
df = pd.read_parquet(ARCHIVO_PARQUET, columns=[image_column])
print(f"Extract {len(df)} images")

for index, row in df.iterrows():
    # 1. get bytes
    image_bytes = row[image_column]

    # 2. check image
    if not image_bytes:
        print(f"Fila {index} no contiene datos de imagen. Saltando.")
        continue

    # 3. output name
    nombre_archivo = f'imagen_{index}.{output_format}'
    ruta_salida = os.path.join(output_folder, nombre_archivo)

    try:
        # 4. write bytes
        with open(ruta_salida, 'wb') as f:
            f.write(image_bytes)

    except Exception as e:
        print(f"Error al procesar o guardar la imagen en la fila {index}: {e}")

print(f"¡Extracción completada! {len(df)} imágenes guardadas en '{output_folder}'.")
