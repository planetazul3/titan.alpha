#!/bin/bash
# .devcontainer/install_talib.sh

# --- PARTE 1: TA-Lib (Sistema y Python) ---
echo "Instalando TA-Lib..."
if [ ! -f "/usr/lib/libta_lib.so" ]; then
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O ta-lib-src.tar.gz
    tar -xzf ta-lib-src.tar.gz
    cd ta-lib
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    rm -rf ta-lib ta-lib-src.tar.gz
else
    echo "TA-Lib ya estaba instalado en el sistema."
fi

# Instala el wrapper de Python (se compila al momento)
pip install ta-lib

# --- PARTE 2: Requirements.txt ---
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
else
    echo "No se encontró requirements.txt, saltando paso."
fi

# --- PARTE 3: Tu librería en Modo Editable ---
echo "Configurando python-deriv-api..."

# Instalamos en modo editable (-e)
pip install -e git+https://github.com/planetazul3/python-deriv-api.git#egg=python-deriv-api