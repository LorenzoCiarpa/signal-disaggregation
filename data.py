from datetime import datetime, timezone

# Convertire secondi Unix in data
timestamp = 1763018881
data = datetime.fromtimestamp(timestamp, tz=timezone.utc)
# print(data)

# Oppure in formato leggibile
print(data.strftime("%Y-%m-%d %H:%M:%S"))