import asyncio
from bleak import BleakScanner

TX_POWER = -59
PATH_LOSS_N = 1.8

TARGET_UUID = "12345678-1234-1234-1234-1234567890AB".upper()

def rssi_to_distance(rssi, tx_power=TX_POWER, n=PATH_LOSS_N):
    """Log-distance path-loss model."""
    return 10 ** ((tx_power - rssi) / (10 * n))


def parse_ibeacon(mfg_data: dict):
    """
    Parse iBeacon manufacturer data.
    Apple Company ID = 0x004C → key 76 in manufacturer_data.
    """
    if 76 not in mfg_data:
        return None

    data = mfg_data[76]
    if len(data) < 23:
        return None

    # 0x02 0x15 header
    if data[0] != 0x02 or data[1] != 0x15:
        return None

    # Extract UUID (16 bytes)
    uuid_bytes = data[2:18]
    uuid = (
        uuid_bytes[0:4].hex() + "-" +
        uuid_bytes[4:6].hex() + "-" +
        uuid_bytes[6:8].hex() + "-" +
        uuid_bytes[8:10].hex() + "-" +
        uuid_bytes[10:16].hex()
    ).upper()

    major = int.from_bytes(data[18:20], "big")
    minor = int.from_bytes(data[20:22], "big")
    tx_power = int.from_bytes(data[22:23], "big", signed=True)

    return uuid, major, minor, tx_power


async def main():
    print("Starting continuous iBeacon scan (Bleak 2.0.0)…")

    # Start scanner using async context manager
    async with BleakScanner() as scanner:
        # advertisement_data() returns an async generator
        async for device, adv in scanner.advertisement_data():
            parsed = parse_ibeacon(adv.manufacturer_data)
            if not parsed:
                continue

            uuid, major, minor, tx_power = parsed

            # Filter your beacons
            if uuid != TARGET_UUID:
                continue

            distance = rssi_to_distance(adv.rssi, tx_power)

            print(
                f"{device.address} | RSSI={adv.rssi:4d} | "
                f"UUID={uuid} | Major={major:3d} | Minor={minor:3d} | "
                f"Dist={distance:5.2f} m"
            )


if __name__ == "__main__":
    asyncio.run(main())