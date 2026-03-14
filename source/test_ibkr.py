"""
test_ibkr.py
------------
Script sencillo para verificar la conexión con Interactive Brokers (TWS o IB Gateway).
Ejecutar desde la terminal para descartar problemas de configuración de TWS.
"""

from ib_insync import IB, Stock, util
import asyncio

async def test_connection():
    ib = IB()
    print("Intentando conectar a TWS/IB Gateway en localhost:7497 (Paper Trading)...")
    try:
        # Intentamos conectar a Paper Trading (7497) por defecto
        # Si prefieres Live, cambia a 7496
        await ib.connectAsync('127.0.0.1', 7497, clientId=99)
        print("✅ CONECTADO EXITOSAMENTE")
        
        print("\nResumen de cuenta:")
        summary = ib.accountSummary()
        for item in summary:
            if item.tag in ('TotalCashValue', 'AvailableFunds', 'NetLiquidation'):
                print(f"  - {item.tag}: {item.value}")
        
        print("\nProbando resolución de contrato (AAPL)...")
        contract = Stock('AAPL', 'SMART', 'USD')
        qualified = ib.qualifyContracts(contract)
        if qualified:
            print(f"  - Contrato verificado: {contract.symbol} ({contract.conId})")
        else:
            print("  - ❌ No se pudo resolver el contrato.")

        ib.disconnect()
        print("\nDesconectado correctamente.")

    except Exception as e:
        print(f"\n❌ ERROR DE CONEXIÓN: {e}")
        print("\nREVISA:")
        print("1. ¿Está TWS o IB Gateway abierto?")
        print("2. ¿La configuración de API tiene activado 'Enable ActiveX and Socket Clients'?")
        print("3. ¿El puerto es 7497 (Paper) o 7496 (Live)?")

if __name__ == "__main__":
    asyncio.run(test_connection())
