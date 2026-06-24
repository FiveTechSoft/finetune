#!/usr/bin/env python3
"""
Harbour/FWH Dataset Improvement Script
Generates synthetic examples for underrepresented topics and instruction types.
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = "Eres un programador experto en Harbour y FiveWin (FWH). Escribe código limpio, optimizado y bien documentado usando variables LOCAL y mandatos nativos de Harbour."

# Templates for low-coverage topics
MULTITHREADING_TEMPLATES = [
    {
        "instruction": "Explica cómo crear un hilo en Harbour usando TThread",
        "output": """Para crear un hilo en Harbour con FiveWin, usamos la clase TThread:

```harbour
#include "fivewin.ch"
#include "thread.ch"

CLASS MyThread FROM TThread
   DATA oParent
   DATA lRunning AS .T.

   METHOD New( oParent )
   METHOD Run()
   METHOD Stop()
ENDCLASS

METHOD New( oParent ) CLASS MyThread
   ::oParent := oParent
   ::super:New()
   RETURN SELF

METHOD Run() CLASS MyThread
   WHILE ::lRunning
      // Trabajo en segundo plano
      PostMessage( ::oParent:hWnd, WM_USER + 1, 0, 0 )
      Sleep( 100 )
   ENDDO
   RETURN NIL

METHOD Stop() CLASS MyThread
   ::lRunning := .F.
   RETURN NIL
```

Uso:
```harbour
oThread := MyThread():New( oMainWindow )
oThread:Run()
```
"""
    },
    {
        "instruction": "Cómo sincronizar acceso a variables compartidas entre hilos",
        "output": """Usa MUTEX para sincronizar acceso:

```harbour
#include "fivewin.ch"

STATIC oMutex := NIL
STATIC nSharedCounter := 0

FUNCTION InitMutex()
   oMutex := TMutex():New()
   RETURN NIL

FUNCTION IncrementCounter()
   LOCAL nValue
   oMutex:Lock()
   nSharedCounter++
   nValue := nSharedCounter
   oMutex:UnLock()
   RETURN nValue

FUNCTION GetCounter()
   LOCAL nValue
   oMutex:Lock()
   nValue := nSharedCounter
   oMutex:UnLock()
   RETURN nValue
```

IMPORTANTE: Siempre usar Lock/UnLock para proteger variables compartidas.
"""
    },
    {
        "instruction": "Crea un pool de hilos para procesar tareas concurrentes",
        "output": """Pool de hilos para procesamiento paralelo:

```harbour
CLASS ThreadPool
   DATA aThreads AS {}
   DATA nMaxThreads AS 5
   DATA oMutex
   DATA aTaskQueue AS {}

   METHOD New( nMax )
   METHOD AddTask( bTask )
   METHOD ProcessNext()
   METHOD Cleanup()
ENDCLASS

METHOD New( nMax ) CLASS ThreadPool
   ::nMaxThreads := nMax
   ::oMutex := TMutex():New()
   RETURN SELF

METHOD AddTask( bTask ) CLASS ThreadPool
   ::oMutex:Lock()
   AAdd( ::aTaskQueue, bTask )
   ::oMutex:UnLock()
   ::ProcessNext()
   RETURN NIL

METHOD ProcessNext() CLASS ThreadPool
   LOCAL bTask
   IF Len( ::aThreads ) < ::nMaxThreads .AND. Len( ::aTaskQueue ) > 0
      ::oMutex:Lock()
      bTask := ATail( ::aTaskQueue )
      ASize( ::aTaskQueue, Len( ::aTaskQueue ) - 1 )
      ::oMutex:UnLock()
      AAdd( ::aThreads, Thread():New( bTask ) )
   ENDIF
   RETURN NIL
```
"""
    }
]

NETWORKING_TEMPLATES = [
    {
        "instruction": "Cómo crear un servidor TCP en Harbour",
        "output": """Servidor TCP básico con FiveWin:

```harbour
#include "fivewin.ch"
#include "tcpclient.ch"

CLASS TCPServer
   DATA nPort
   DATA lRunning AS .F.
   DATA oSocket

   METHOD New( nPort )
   METHOD Start()
   METHOD Stop()
   METHOD OnAccept( oSocket )
   METHOD OnData( oSocket, cData )
ENDCLASS

METHOD New( nPort ) CLASS TCPServer
   ::nPort := nPort
   RETURN SELF

METHOD Start() CLASS TCPServer
   ::oSocket := TServerSocket():New( ::nPort )
   ::oSocket:bOnAccept := {|o| ::OnAccept(o) }
   ::lRunning := .T.
   ::oSocket:Listen()
   RETURN NIL

METHOD OnAccept( oSocket ) CLASS TCPServer
   local oClient := oSocket:Accept()
   oClient:bOnData := {|o,c| ::OnData(o,c) }
   RETURN NIL

METHOD OnData( oSocket, cData ) CLASS TCPServer
   // Procesar datos recibidos
   ? "Recibido:", cData
   oSocket:Send( "OK: " + cData )
   RETURN NIL
```
"""
    },
    {
        "instruction": "Realizar una petición HTTP GET en Harbour",
        "output": """Para peticiones HTTP usamos la librería THttpClient:

```harbour
#include "fivehuff.ch"

FUNCTION HttpGet( cUrl )
   LOCAL oHttp, cResponse
   
   oHttp := THttpClient():New()
   oHttp:SetUrl( cUrl )
   oHttp:SetTimeout( 30 )
   
   IF oHttp:Execute( "GET" )
      cResponse := oHttp:GetResponse()
   ELSE
      cResponse := "Error: " + oHttp:GetError()
   ENDIF
   
   oHttp:Close()
   RETURN cResponse
```

Uso:
```harbour
cData := HttpGet( "https://api.example.com/data" )
? cData
```
"""
    },
    {
        "instruction": "Crear un cliente WebSocket en Harbour",
        "output": """Cliente WebSocket con FiveWin:

```harbour
CLASS WebSocketClient
   DATA oSocket
   DATA cUrl
   DATA bOnMessage
   DATA bOnOpen
   DATA bOnClose

   METHOD New( cUrl )
   METHOD Connect()
   METHOD Send( cMessage )
   METHOD Close()
ENDCLASS

METHOD New( cUrl ) CLASS WebSocketClient
   ::cUrl := cUrl
   RETURN SELF

METHOD Connect() CLASS WebSocketClient
   ::oSocket := TWebSocket():New( ::cUrl )
   ::oSocket:bOnOpen := ::bOnOpen
   ::oSocket:bOnMessage := ::bOnMessage
   ::oSocket:bOnClose := ::bOnClose
   ::oSocket:Connect()
   RETURN NIL

METHOD Send( cMessage ) CLASS WebSocketClient
   ::oSocket:Send( cMessage )
   RETURN NIL

METHOD Close() CLASS WebSocketClient
   ::oSocket:Close()
   RETURN NIL
```
"""
    }
]

ERROR_HANDLING_TEMPLATES = [
    {
        "instruction": "Implementar manejo de errores con BEGIN SEQUENCE en Harbour",
        "output": """BEGIN SEQUENCE es el mecanismo nativo de Harbour para manejo de errores:

```harbour
FUNCTION SafeOperation()
   LOCAL nResult
   
   BEGIN SEQUENCE
      nResult := 10 / 0  // Error de división por cero
      ? "Resultado:", nResult
   
   RECOVER USING oError
      ? "Error capturado:", oError:description
      ? "Archivo:", oError:filename
      ? "Línea:", oError:lineno
      nResult := 0
   
   END SEQUENCE
   
   RETURN nResult
```

Con THROW para errores personalizados:
```harbour
BEGIN SEQUENCE
   IF Empty( cData )
      THROW( ErrorNew( "MYERROR", 0, "Datos vacíos" ) )
   ENDIF
   // Procesar datos
RECOVER USING oError
   HandleError( oError )
END SEQUENCE
```
"""
    },
    {
        "instruction": "Crear una función de logging de errores robusta",
        "output": """Sistema de logging profesional:

```harbour
CLASS ErrorLogger
   DATA cLogFile
   DATA oMutex

   METHOD New( cLogFile )
   METHOD LogError( oError )
   METHOD LogMessage( cMessage, cLevel )
   METHOD GetTimestamp()
ENDCLASS

METHOD New( cLogFile ) CLASS ErrorLogger
   ::cLogFile := cLogFile
   ::oMutex := TMutex():New()
   RETURN SELF

METHOD LogError( oError ) CLASS ErrorLogger
   LOCAL cMsg
   cMsg := ::GetTimestamp() + " [ERROR] " + ;
           oError:description + " in " + oError:filename + ;
           " line " + AllTrim( Str( oError:lineno ) )
   ::LogMessage( cMsg, "ERROR" )
   RETURN NIL

METHOD LogMessage( cMessage, cLevel ) CLASS ErrorLogger
   LOCAL hFile
   ::oMutex:Lock()
   hFile := FOpen( ::cLogFile, FO_WRITE )
   FSeek( hFile, 0, FS_END )
   FWrite( hFile, cMessage + HB_EOL() )
   FClose( hFile )
   ::oMutex:UnLock()
   RETURN NIL

METHOD GetTimestamp() CLASS ErrorLogger
   RETURN DTOS( Date() ) + " " + Time()
```
"""
    },
    {
        "instruction": "Cómo implementar retry con backoff exponencial",
        "output": """Patrón retry con backoff exponencial:

```harbour
FUNCTION RetryWithBackoff( bOperation, nMaxRetries, nBaseDelay )
   LOCAL nAttempt, nDelay, lSuccess, oError
   
   DEFAULT nMaxRetries TO 3
   DEFAULT nBaseDelay TO 1000  // 1 segundo
   
   nAttempt := 0
   lSuccess := .F.
   
   WHILE !lSuccess .AND. nAttempt < nMaxRetries
      BEGIN SEQUENCE
         EVAL( bOperation )
         lSuccess := .T.
      RECOVER USING oError
         nAttempt++
         nDelay := nBaseDelay * (2 ^ (nAttempt - 1))
         ? "Intento", nAttempt, "falló:", oError:description
         ? "Reintentando en", nDelay, "ms..."
         Sleep( nDelay )
      END SEQUENCE
   ENDDO
   
   IF !lSuccess
      THROW( ErrorNew( "RETRY", 0, "Máximo de reintentos alcanzado" ) )
   ENDIF
   
   RETURN NIL
```

Uso:
```harbour
RetryWithBackoff( {|| ConnectToServer() }, 5, 500 )
```
"""
    }
]

DATABASE_TEMPLATES = [
    {
        "instruction": "Conectar y consultar una base de datos DBF en Harbour",
        "output": """Operaciones con DBF usando RDD nativo:

```harbour
#include "dbinfo.ch"

FUNCTION QueryDBF( cDbfPath, cFilter )
   LOCAL nArea
   
   // Abrir archivo DBF
   nArea := select()
   USE ( cDbfPath ) ALIAS "myfile" SHARED
   
   IF neterr()
      ? "Error al abrir archivo"
      RETURN NIL
   ENDIF
   
   // Aplicar filtro
   IF !Empty( cFilter )
      SET FILTER TO &(cFilter)
      GO TOP
   ENDIF
   
   // Recorrer registros
   WHILE !EOF()
      ? FIELDGET(1), FIELDGET(2), FIELDGET(3)
      SKIP
   ENDDO
   
   // Cerrar archivo
   USE
   SELECT ( nArea )
   
   RETURN NIL
```
"""
    },
    {
        "instruction": "Realizar transacciones SQL con Firebird/PostgreSQL",
        "output": """Conexión SQL con FiveWin:

```harbour
#include "fivewin.ch"
#include "dbfddox.ch"

FUNCTION SQLTransaction()
   LOCAL oDB, lSuccess := .F.
   
   // Conectar a la base de datos
   oDB := TSQLConnection():New( ;
      "FBDB", ;           // Driver
      "localhost", ;      // Servidor
      "mydb", ;           // Base de datos
      "SYSDBA", ;         // Usuario
      "masterkey" )       // Password
   
   IF oDB:Connect()
      oDB:BeginTransaction()
      
      BEGIN SEQUENCE
         oDB:Execute( "INSERT INTO clientes (nombre) VALUES ('Juan')" )
         oDB:Execute( "UPDATE productos SET stock = stock - 1 WHERE id = 1" )
         oDB:Commit()
         lSuccess := .T.
      RECOVER USING oError
         oDB:Rollback()
         ? "Error:", oError:description
      END SEQUENCE
      
      oDB:Disconnect()
   ENDIF
   
   RETURN lSuccess
```
"""
    }
]

OOP_TEMPLATES = [
    {
        "instruction": "Explica el modelo de herencia en Harbour/OOP",
        "output": """Herencia en Harbour/OOP:

```harbour
// Clase base
CLASS Animal
   DATA cName
   DATA nAge
   
   METHOD New( cName, nAge )
   METHOD Speak()
   METHOD Describe()
ENDCLASS

METHOD New( cName, nAge ) CLASS Animal
   ::cName := cName
   ::nAge := nAge
   RETURN SELF

METHOD Speak() CLASS Animal
   RETURN "..."

METHOD Describe() CLASS Animal
   RETURN ::cName + " tiene " + AllTrim( Str( ::nAge ) ) + " años"

// Clase derivada
CLASS Dog FROM Animal
   DATA cBreed
   
   METHOD New( cName, nAge, cBreed )
   METHOD Speak()
   METHOD Fetch()
ENDCLASS

METHOD New( cName, nAge, cBreed ) CLASS Dog
   ::super:New( cName, nAge )
   ::cBreed := cBreed
   RETURN SELF

METHOD Speak() CLASS Dog
   RETURN "Guau!"

METHOD Fetch() CLASS Dog
   RETURN ::cName + " trae la pelota"
```

Uso:
```harbour
oDog := Dog():New("Max", 3, "Labrador")
? oDog:Speak()    // "Guau!"
? oDog:Describe()  // "Max tiene 3 años"
```
"""
    },
    {
        "instruction": "Implementar interfaz (mixin) en Harbour",
        "output": """Patrón de mixin en Harbour (simulado):

```harbour
// Interfaz de serialización
CLASS Serializable
   METHOD Serialize()
   METHOD Deserialize( cData )
ENDCLASS

METHOD Serialize() CLASS Serializable
   // Implementación por defecto
   RETURN HB_Serialize( ::GetFields() )

METHOD Deserialize( cData ) CLASS Serializable
   LOCAL aData := HB_Deserialize( cData )
   ::SetFields( aData )
   RETURN NIL

// Clase que usa mixin
CLASS User FROM Serializable
   DATA cName
   DATA cEmail
   
   METHOD New( cName, cEmail )
   METHOD GetFields()
   METHOD SetFields( aData )
ENDCLASS

METHOD New( cName, cEmail ) CLASS User
   ::cName := cName
   ::cEmail := cEmail
   RETURN SELF

METHOD GetFields() CLASS User
   RETURN { ::cName, ::cEmail }

METHOD SetFields( aData ) CLASS User
   ::cName := aData[1]
   ::cEmail := aData[2]
   RETURN NIL
```
"""
    }
]

FIX_TEMPLATES = [
    {
        "instruction": "Mi código DBF no cierra correctamente, ¿qué está mal?",
        "output": """Problema común: no se verifica el estado antes de cerrar.

MAL:
```harbour
USE myfile
// ... operaciones
USE  // Cierra todos los archivos
```

BIEN:
```harbour
USE myfile
// ... operaciones
IF Used()
   USE
ENDIF
```

Si usas áreas múltiples:
```harbour
SELECT 1
USE archivo1
SELECT 2
USE archivo2

// Al cerrar:
IF Used(1)
   SELECT 1
   USE
ENDIF
IF Used(2)
   SELECT 2
   USE
ENDIF
```
"""
    },
    {
        "instruction": "El filtro SET FILTER TO no funciona, ¿por qué?",
        "output": """Causas comunes de SET FILTER TO:

1. **Expresión inválida:**
```harbour
// MAL - falta & para macro
SET FILTER TO estado = "A"

// BIEN - usar macro
SET FILTER TO estado = "A"

// O mejor - usar FOR
SET FILTER TO estado == "A"
```

2. **No se ejecuta GO TOP después:**
```harbour
SET FILTER TO estado = "A"
GO TOP  // ¡Imprescindible!
```

3. **Archivo no está abierto:**
```harbour
IF Used()
   SET FILTER TO estado = "A"
   GO TOP
ENDIF
```

4. **Usar DbSetFilter() para mejor rendimiento:**
```harbour
DbSetFilter( {|| estado == "A" }, 'estado == "A"' )
GO TOP
```
"""
    }
]

def load_existing_dataset(filepath):
    """Load existing dataset"""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def create_example(instruction, output, system=SYSTEM_PROMPT):
    """Create a dataset example"""
    return {
        "system": system,
        "instruction": instruction,
        "input": "",
        "output": output
    }

def augment_examples(examples):
    """Create variations of existing examples"""
    augmented = []
    
    # Add "explain" type questions
    explain_patterns = [
        "Explica cómo funciona {topic}",
        "¿Qué hace这段 código de Harbour?",
        "Describe el propósito de esta función",
        "¿Por qué se usa {pattern} en Harbour?"
    ]
    
    # Add "fix" type questions
    fix_patterns = [
        "Mi código no funciona, ¿qué está mal?",
        "Obtengo este error: {error}, ¿cómo lo arreglo?",
        "¿Cuál es el problema con este código?",
        "Optimiza esta función que es muy lenta"
    ]
    
    return augmented

def main():
    print("🚀 Mejorando dataset Harbour/FWH...")
    
    # Load existing dataset
    examples = load_existing_dataset('/home/antonio/finetune/harbour_fwh_dedup.jsonl')
    print(f"📥 Cargados {len(examples)} ejemplos existentes")
    
    new_examples = []
    
    # Add multithreading examples
    print("🧵 Añadiendo ejemplos de multithreading...")
    for template in MULTITHREADING_TEMPLATES:
        new_examples.append(create_example(template['instruction'], template['output']))
    
    # Add networking examples
    print("🌐 Añadiendo ejemplos de networking...")
    for template in NETWORKING_TEMPLATES:
        new_examples.append(create_example(template['instruction'], template['output']))
    
    # Add error handling examples
    print("⚠️  Añadiendo ejemplos de manejo de errores...")
    for template in ERROR_HANDLING_TEMPLATES:
        new_examples.append(create_example(template['instruction'], template['output']))
    
    # Add database examples
    print("🗄️  Añadiendo ejemplos de base de datos...")
    for template in DATABASE_TEMPLATES:
        new_examples.append(create_example(template['instruction'], template['output']))
    
    # Add OOP examples
    print("📦 Añadiendo ejemplos de OOP...")
    for template in OOP_TEMPLATES:
        new_examples.append(create_example(template['instruction'], template['output']))
    
    # Add fix/debug examples
    print("🔧 Añadiendo ejemplos de debugging...")
    for template in FIX_TEMPLATES:
        new_examples.append(create_example(template['instruction'], template['output']))
    
    # Combine and deduplicate
    all_examples = examples + new_examples
    seen = set()
    unique = []
    
    for ex in all_examples:
        key = ex['instruction']
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    
    print(f"\n📊 Total original: {len(examples)}")
    print(f"📊 Nuevos añadidos: {len(new_examples)}")
    print(f"📊 Total final (sin duplicados): {len(unique)}")
    
    # Save improved dataset
    output_path = '/home/antonio/finetune/harbour_fwh_improved.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in unique:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Guardado en: {output_path}")
    print(f"📈 Mejora: +{len(unique) - len(examples)} ejemplos ({((len(unique) - len(examples)) / len(examples) * 100):.1f}% más)")

if __name__ == "__main__":
    main()
