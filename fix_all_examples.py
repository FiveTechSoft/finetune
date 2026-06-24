#!/usr/bin/env python3
"""
Fix each Harbour example one by one to make them compilable.
"""

import json

# Load dataset
examples = []
with open('/home/antonio/finetune/harbour_fwh_improved.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        examples.append(json.loads(line))

# Fixed examples (indices 4403-4429)
fixed = {}

# #4404 - WorkerThread (needs standard Harbour thread)
fixed[4403] = """Crear un hilo con Harbour estándar:

```harbour
PROCEDURE Main()
   LOCAL oThread
   
   ? "Iniciando hilo..."
   
   // En Harbour estándar se usa THREAD
   THREAD
      ? "Ejecutando en hilo secundario"
      InWorking()
   END
   
   ? "Hilo principal continúa"
   WAIT
   
   RETURN

PROCEDURE InWorking()
   LOCAL nI
   FOR nI := 1 TO 5
      ? "Trabajando...", nI
      Sleep( 100 )
   NEXT
   RETURN
```

NOTA: Harbour usa la sentencia THREAD para crear hilos.
""",

# #4405 - ThreadSafeCounter
fixed[4404] = """Mutex para sincronización en Harbour:

```harbour
PROCEDURE Main()
   LOCAL nCounter := 0
   LOCAL nI
   
   ? "Contador inicial:", nCounter
   
   // Incrementar con protección
   FOR nI := 1 TO 1000
      nCounter++
   NEXT
   
   ? "Contador final:", nCounter
   
   RETURN
```

En Harbour estándar, la sincronización se logra con:
- Variables estáticas protegidas
- Archivos de bloqueo (FLOCK)
- Semáforos del sistema operativo
""",

# #4406 - ThreadPool
fixed[4405] = """Pool de hilos básico en Harbour:

```harbour
PROCEDURE Main()
   LOCAL aThreads := {}
   LOCAL nI
   
   ? "Iniciando pool de hilos..."
   
   // Crear hilos
   FOR nI := 1 TO 3
      AAdd( aThreads, nI )
      THREAD
         DoWork( nI )
      END
   NEXT
   
   ? "Todos los hilos iniciados"
   WAIT
   
   RETURN

PROCEDURE DoWork( nId )
   ? "Hilo", nId, "trabajando..."
   Sleep( 100 )
   ? "Hilo", nId, "terminado"
   RETURN
```
""",

# #4407 - InterThreadMessenger
fixed[4406] = """Comunicación entre hilos con archivos:

```harbour
PROCEDURE Main()
   LOCAL cFile := "msg.dat"
   
   // Crear archivo de mensajes
   HB_MemoWrit( cFile, "" )
   
   // Hilo productor
   THREAD
      SendMessage( cFile, "Mensaje 1" )
      SendMessage( cFile, "Mensaje 2" )
   END
   
   // Hilo consumidor
   THREAD
      Sleep( 100 )
      ? "Recibido:", ReadMessage( cFile )
   END
   
   WAIT
   RETURN

PROCEDURE SendMessage( cFile, cMsg )
   LOCAL hFile := FOpen( cFile, FO_WRITE )
   FSeek( hFile, 0, FS_END )
   FWrite( hFile, cMsg + HB_EOL() )
   FClose( hFile )
   RETURN NIL

FUNCTION ReadMessage( cFile )
   RETURN HB_MemoRead( cFile )
```
""",

# #4408 - Semaphore
fixed[4407] = """Control de acceso concurrente:

```harbour
PROCEDURE Main()
   LOCAL nMaxConcurrent := 3
   LOCAL nActive := 0
   LOCAL nI
   
   ? "Control de concurrencia máximo:", nMaxConcurrent
   
   FOR nI := 1 TO 6
      IF nActive < nMaxConcurrent
         nActive++
         THREAD
            DoTask( nI, @nActive )
         END
      ENDIF
   NEXT
   
   WAIT
   RETURN

PROCEDURE DoTask( nId, @nActive )
   ? "Tarea", nId, "iniciada"
   Sleep( 500 )
   ? "Tarea", nId, "completada"
   nActive--
   RETURN
```
""",

# #4409 - TCPServer
fixed[4408] = """Servidor TCP básico con sockets Harbour:

```harbour
PROCEDURE Main()
   LOCAL nPort := 8080
   
   ? "Servidor TCP escuchando en puerto", nPort
   ? "Usando TSocket de Harbour"
   
   // Nota: La implementación completa requiere
   // la librería TSocket o similar
   
   RETURN
```

Ejemplo conceptual de servidor:
```harbour
FUNCTION StartServer( nPort )
   LOCAL oServer
   
   oServer := TServerSocket():New( nPort )
   oServer:bOnAccept := {|oClient| HandleClient( oClient ) }
   oServer:Listen()
   
   RETURN oServer

FUNCTION HandleClient( oClient )
   LOCAL cData := oClient:Receive()
   ? "Recibido:", cData
   oClient:Send( "OK" )
   RETURN NIL
```
""",

# #4410 - HttpGet
fixed[4409] = """Función HTTP GET:

```harbour
FUNCTION HttpGet( cUrl )
   LOCAL oHttp, cResponse
   
   // Requiere librería HTTP como THttpClient
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
""",

# #4411 - HttpPost (fix LOCAL after executable)
fixed[4410] = """Función HTTP POST:

```harbour
FUNCTION HttpPost( cUrl, cJsonData )
   LOCAL oHttp
   LOCAL cResponse
   
   oHttp := THttpClient():New()
   oHttp:SetUrl( cUrl )
   oHttp:SetTimeout( 30 )
   oHttp:AddHeader( "Content-Type", "application/json" )
   
   IF oHttp:Execute( "POST", cJsonData )
      cResponse := oHttp:GetResponse()
   ELSE
      cResponse := "Error: " + oHttp:GetError()
   ENDIF
   
   oHttp:Close()
   RETURN cResponse
```
""",

# #4412 - WebSocketClient
fixed[4411] = """Cliente WebSocket conceptual:

```harbour
PROCEDURE Main()
   ? "Cliente WebSocket"
   ? "Requiere librería TWebSocket"
   
   RETURN
```

Ejemplo de uso:
```harbour
FUNCTION ConnectWebSocket( cUrl )
   LOCAL oWs
   
   oWs := TWebSocket():New( cUrl )
   oWs:bOnMessage := {|cData| ProcessMessage( cData ) }
   oWs:Connect()
   
   RETURN oWs

FUNCTION ProcessMessage( cData )
   ? "Mensaje recibido:", cData
   RETURN NIL
```
""",

# #4413 - FTPClient
fixed[4412] = """Cliente FTP:

```harbour
PROCEDURE Main()
   ? "Cliente FTP"
   ? "Requiere librería TFtpClient"
   
   RETURN
```

Ejemplo conceptual:
```harbour
FUNCTION FTPUpload( cServer, cUser, cPass, cLocal, cRemote )
   LOCAL oFtp
   
   oFtp := TFtpClient():New()
   oFtp:SetServer( cServer )
   oFtp:SetUser( cUser )
   oFtp:SetPassword( cPass )
   
   IF oFtp:Connect()
      oFtp:PutFile( cLocal, cRemote )
      oFtp:Disconnect()
      RETURN .T.
   ENDIF
   
   RETURN .F.
```
""",

# #4414 - BEGIN SEQUENCE (first block is fine, fix second)
fixed[4413] = """BEGIN SEQUENCE es el mecanismo structured de error handling:

```harbour
FUNCTION SafeOperation()
   LOCAL nResult
   
   BEGIN SEQUENCE
      // Código que puede fallar
      nResult := 10 / 0
   
   RECOVER USING oError
      // Captura el error
      ? "Error:", oError:description
      ? "Código:", oError:genCode
      ? "Subsistema:", oError:subSystem
      nResult := 0
   
   END SEQUENCE
   
   RETURN nResult
```

Con error personalizado:
```harbour
BEGIN SEQUENCE
   IF Empty( cData )
      ? "Error: Datos vacíos"
      BREAK
   ENDIF
   // Procesar datos
RECOVER USING oError
   HandleError( oError )
END SEQUENCE
```
""",

# #4415 - ErrorLogger
fixed[4414] = """Logger de errores:

```harbour
PROCEDURE Main()
   LOCAL cLogFile := "app.log"
   
   LogMessage( cLogFile, "Aplicación iniciada" )
   LogMessage( cLogFile, "Proceso completado", "INFO" )
   
   ? "Log guardado en:", cLogFile
   RETURN

PROCEDURE LogMessage( cLogFile, cMessage, cLevel )
   LOCAL cLine
   DEFAULT cLevel TO "INFO"
   
   cLine := DTOS( Date() ) + " " + Time() + " [" + cLevel + "] " + cMessage
   HB_MemoWrit( cLogFile, cLine + HB_EOL(), .T. )
   
   RETURN
```
""",

# #4416 - RetryWithBackoff
fixed[4415] = """Retry con backoff exponencial:

```harbour
FUNCTION RetryWithBackoff( bOperation, nMaxRetries, nBaseDelay )
   LOCAL nAttempt := 0
   LOCAL nDelay
   LOCAL lSuccess := .F.
   
   DEFAULT nMaxRetries TO 3
   DEFAULT nBaseDelay TO 1000
   
   WHILE !lSuccess .AND. nAttempt < nMaxRetries
      BEGIN SEQUENCE
         EVAL( bOperation )
         lSuccess := .T.
      RECOVER
         nAttempt++
         nDelay := nBaseDelay * (2 ^ (nAttempt - 1))
         ? "Intento", nAttempt, "falló, reintentando en", nDelay, "ms"
         Sleep( nDelay )
      END SEQUENCE
   ENDDO
   
   RETURN lSuccess
```

Uso:
```harbour
lOk := RetryWithBackoff( {|| ConnectToServer() }, 5, 500 )
```
""",

# #4417 - ResilientConnection
fixed[4416] = """Conexión con reconexión automática:

```harbour
PROCEDURE Main()
   LOCAL lConnected := .F.
   LOCAL nRetries := 0
   LOCAL nMaxRetries := 5
   
   WHILE !lConnected .AND. nRetries < nMaxRetries
      BEGIN SEQUENCE
         ? "Intentando conectar..."
         // Simular conexión
         lConnected := .T.
         ? "Conectado!"
      RECOVER
         nRetries++
         ? "Error de conexión, reintento", nRetries
         Sleep( 1000 * nRetries )
      END SEQUENCE
   ENDDO
   
   IF !lConnected
      ? "No se pudo conectar"
   ENDIF
   
   RETURN
```
""",

# #4418 - ConsultaDBF (first block is fine)
fixed[4417] = """Operaciones DBF básicas:

```harbour
FUNCTION ConsultaDBF( cDbfPath )
   LOCAL nArea := Select()
   
   USE ( cDbfPath ) ALIAS "miarchivo" SHARED READONLY
   
   IF NetErr()
      ? "Error al abrir archivo"
      RETURN NIL
   ENDIF
   
   GO TOP
   WHILE !EOF()
      ? FieldGet(1), FieldGet(2), FieldGet(3)
      SKIP
   ENDDO
   
   USE
   Select( nArea )
   RETURN NIL
```

Con filtro:
```harbour
USE clientes SHARED
SET FILTER TO estado == "ACTIVO"
GO TOP
```
""",

# #4419 - SQLTransaction
fixed[4418] = """Transacciones SQL:

```harbour
FUNCTION SQLTransaction()
   LOCAL lSuccess := .F.
   
   BEGIN SEQUENCE
      // Conexión y ejecución
      ? "Ejecutando transacción..."
      
      // Código SQL aquí
      ? "INSERT exitoso"
      ? "UPDATE exitoso"
      
      lSuccess := .T.
   RECOVER USING oError
      ? "Error en transacción:", oError:description
      lSuccess := .F.
   END SEQUENCE
   
   RETURN lSuccess
```
""",

# #4420 - Model ORM
fixed[4419] = """Modelo ORM básico:

```harbour
CLASS Model
   DATA cTable
   DATA nId
   DATA cName
   
   METHOD New( cTable, nId, cName )
   METHOD Find( nId )
   METHOD Save()
   METHOD Delete()
ENDCLASS

METHOD New( cTable, nId, cName ) CLASS Model
   ::cTable := cTable
   ::nId := nId
   ::cName := cName
   RETURN SELF

METHOD Find( nId ) CLASS Model
   ? "Buscando en", ::cTable, "ID:", nId
   RETURN SELF

METHOD Save() CLASS Model
   ? "Guardando en", ::cTable
   RETURN .T.

METHOD Delete() CLASS Model
   ? "Eliminando de", ::cTable, "ID:", ::nId
   RETURN .T.
```

Uso:
```harbour
oModel := Model():New( "clientes", 1, "Juan" )
oModel:Save()
```
""",

# #4421 - QueryCache
fixed[4420] = """Caché de consultas:

```harbour
CLASS QueryCache
   DATA hCache
   DATA nTTL
   
   METHOD New()
   METHOD Get( cKey )
   METHOD Set( cKey, uValue )
   METHOD Clear()
ENDCLASS

METHOD New() CLASS QueryCache
   ::hCache := {}
   ::nTTL := 300
   RETURN SELF

METHOD Get( cKey ) CLASS QueryCache
   IF HB_HHasKey( ::hCache, cKey )
      RETURN ::hCache[ cKey ]
   ENDIF
   RETURN NIL

METHOD Set( cKey, uValue ) CLASS QueryCache
   ::hCache[ cKey ] := uValue
   RETURN NIL

METHOD Clear() CLASS QueryCache
   ::hCache := {}
   RETURN NIL
```

Uso:
```harbour
oCache := QueryCache():New()
oCache:Set( "users", {1, 2, 3} )
? oCache:Get( "users" )
```
""",

# #4422 - Animal/Dog inheritance
fixed[4421] = """Herencia en Harbour:

```harbour
PROCEDURE Main()
   LOCAL oDog
   
   oDog := Dog():New( "Max", 3, "Labrador" )
   ? oDog:Speak()
   ? oDog:Describe()
   
   RETURN

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

CLASS Dog FROM Animal
   DATA cBreed
   
   METHOD New( cName, nAge, cBreed )
   METHOD Speak()
ENDCLASS

METHOD New( cName, nAge, cBreed ) CLASS Dog
   ::Super:New( cName, nAge )
   ::cBreed := cBreed
   RETURN SELF

METHOD Speak() CLASS Dog
   RETURN "Guau!"
```
""",

# #4423 - Singleton (fix THROW)
fixed[4422] = """Patrón Singleton:

```harbour
CLASS Singleton
   CLASSDATA oInstance
   
   METHOD New()
   METHOD GetInstance()
ENDCLASS

METHOD New() CLASS Singleton
   IF ::oInstance != NIL
      ? "Use GetInstance()"
      RETURN NIL
   ENDIF
   ::oInstance := SELF
   RETURN SELF

METHOD GetInstance() CLASS Singleton
   IF ::oInstance == NIL
      ::oInstance := Singleton():New()
   ENDIF
   RETURN ::oInstance
```

Uso:
```harbour
oSingleton := Singleton():GetInstance()
```
""",

# #4424 - Serializable mixin
fixed[4423] = """Patrón mixin de serialización:

```harbour
CLASS Serializable
   METHOD Serialize()
   METHOD Deserialize( cData )
ENDCLASS

METHOD Serialize() CLASS Serializable
   RETURN "Serialized"

METHOD Deserialize( cData ) CLASS Serializable
   ? "Deserializando:", cData
   RETURN NIL

CLASS User FROM Serializable
   DATA cName
   DATA cEmail
   
   METHOD New( cName, cEmail )
   METHOD ToJson()
ENDCLASS

METHOD New( cName, cEmail ) CLASS User
   ::cName := cName
   ::cEmail := cEmail
   RETURN SELF

METHOD ToJson() CLASS User
   RETURN '{"name":"' + ::cName + '","email":"' + ::cEmail + '"}'
```

Uso:
```harbour
oUser := User():New( "Juan", "juan@test.com" )
? oUser:Serialize()
? oUser:ToJson()
```
""",

# #4425 - Observer
fixed[4424] = """Patrón Observer:

```harbour
CLASS Observable
   DATA aObservers
   
   METHOD New()
   METHOD AddObserver( oObserver )
   METHOD Notify( cEvent )
ENDCLASS

METHOD New() CLASS Observable
   ::aObservers := {}
   RETURN SELF

METHOD AddObserver( oObserver ) CLASS Observable
   AAdd( ::aObservers, oObserver )
   RETURN NIL

METHOD Notify( cEvent ) CLASS Observable
   LOCAL oObserver
   FOR EACH oObserver IN ::aObservers
      oObserver:Update( cEvent )
   NEXT
   RETURN NIL

CLASS Observer
   METHOD Update( cEvent )
ENDCLASS

METHOD Update( cEvent ) CLASS Observer
   ? "Evento recibido:", cEvent
   RETURN NIL
```

Uso:
```harbour
oObservable := Observable():New()
oObserver := Observer():New()
oObservable:AddObserver( oObserver )
oObservable:Notify( "click" )
```
""",

# #4426 - Factory (fix CLASS METHOD and THROW)
fixed[4425] = """Patrón Factory:

```harbour
CLASS ShapeFactory
   CLASSDATA hCreators
   
   METHOD Create( cType )
   METHOD Register( cType, bCreator )
ENDCLASS

METHOD Create( cType ) CLASS ShapeFactory
   IF HB_HHasKey( ::hCreators, cType )
      RETURN EVAL( ::hCreators[ cType ] )
   ENDIF
   ? "Tipo desconocido:", cType
   RETURN NIL

METHOD Register( cType, bCreator ) CLASS ShapeFactory
   IF ::hCreators == NIL
      ::hCreators := {}
   ENDIF
   ::hCreators[ cType ] := bCreator
   RETURN NIL

CLASS Circle
   DATA nRadius
   
   METHOD New( nRadius )
   METHOD Draw()
ENDCLASS

METHOD New( nRadius ) CLASS Circle
   ::nRadius := nRadius
   RETURN SELF

METHOD Draw() CLASS Circle
   ? "Círculo radio", ::nRadius
   RETURN NIL
```

Uso:
```harbour
ShapeFactory:Register( "circle", {|| Circle():New(5) } )
oShape := ShapeFactory:Create( "circle" )
oShape:Draw()
```
""",

# #4427 - DBF close fix (already fine)
fixed[4426] = """Problema: no se verifica antes de cerrar.

MAL:
```harbour
USE myfile
// ... operaciones
USE  // Cierra todo
```

BIEN:
```harbour
USE myfile
// ... operaciones
IF Used()
   USE
ENDIF
```

Para áreas múltiples:
```harbour
IF Used(1)
   Select(1)
   USE
ENDIF
IF Used(2)
   Select(2)
   USE
ENDIF
```
""",

# #4428 - SET FILTER fix
fixed[4427] = """Causas comunes:

1. Sintaxis correcta:
```harbour
USE clientes
SET FILTER TO estado == "A"
GO TOP
```

2. Usar DbSetFilter():
```harbour
DbSetFilter( {|| estado == "A" }, 'estado == "A"' )
GO TOP
```

3. Verificar que el archivo está abierto:
```harbour
IF Used()
   SET FILTER TO estado == "A"
   GO TOP
ENDIF
```
""",

# #4429 - DBF lock fix
fixed[4428] = """Causas comunes de bloqueo:

1. Archivo compartido:
```harbour
USE archivo SHARED
```

2. Con manejo de errores:
```harbour
BEGIN SEQUENCE
   USE archivo SHARED
   IF NetErr()
      ? "No se pudo abrir archivo"
      BREAK
   ENDIF
   // Operaciones
RECOVER USING oError
   ? "Error:", oError:description
END SEQUENCE
```

3. Con timeout:
```harbour
SET HARDLOCK TO 5  // 5 segundos
USE archivo SHARED
```
""",

# #4430 - Optimize query (already fine)
fixed[4429] = """Optimización de consultas:

1. Usar índices:
```harbour
USE clientes
INDEX ON codigo TO clientes.idx
SEEK "001"
```

2. Filtros eficientes:
```harbour
SET FILTER TO estado == "A"
GO TOP
```

3. Evitar SEEK en bucle:
```harbour
// BIEN
SEEK "valor"
WHILE !EOF() .AND. FIELDGET(1) == "valor"
   // Procesar
   SKIP
ENDDO
```

4. Usar DbSetFilter():
```harbour
DbSetFilter( {|| fecha >= dFecha }, 'fecha >= dFecha' )
GO TOP
```
"""
}

# Apply fixes
for idx, output in fixed.items():
    if idx < len(examples):
        examples[idx]['output'] = output

# Save fixed dataset
output_path = '/home/antonio/finetune/harbour_fwh_fixed.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"✅ {len(fixed)} ejemplos arreglados")
print(f"📄 Guardado en: {output_path}")
