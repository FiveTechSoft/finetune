#!/usr/bin/env python3
"""
Enhanced Harbour/FWH Dataset Generator
Creates comprehensive synthetic examples for underrepresented topics.
"""

import json
import random

SYSTEM_PROMPT = "Eres un programador experto en Harbour y FiveWin (FWH). Escribe código limpio, optimizado y bien documentado usando variables LOCAL y mandatos nativos de Harbour."

ENGLISH_SYSTEM = "You are an expert Harbour and FiveWin (FWH) programmer. Write clean, optimized code using LOCAL variables and native Harbour commands."

def create_example(instruction, output, lang="es"):
    system = SYSTEM_PROMPT if lang == "es" else ENGLISH_SYSTEM
    return {
        "system": system,
        "instruction": instruction,
        "input": "",
        "output": output
    }

# Multithreading examples
MULTITHREADING = [
    create_example(
        "Cómo crear un hilo básico en Harbour con TThread",
        """Crear un hilo con TThread:

```harbour
#include "fivewin.ch"
#include "thread.ch"

CLASS WorkerThread FROM TThread
   DATA bWork
   DATA lActive AS .T.

   METHOD New( bWork )
   METHOD Run()
   METHOD Stop()
   METHOD IsActive()
ENDCLASS

METHOD New( bWork ) CLASS WorkerThread
   ::bWork := bWork
   ::super:New()
   RETURN SELF

METHOD Run() CLASS WorkerThread
   WHILE ::lActive
      EVAL( ::bWork )
      Sleep( 10 )
   ENDDO
   RETURN NIL

METHOD Stop() CLASS WorkerThread
   ::lActive := .F.
   RETURN NIL

METHOD IsActive() CLASS WorkerThread
   RETURN ::lActive
```

Uso:
```harbour
oThread := WorkerThread():New( {|| DoWork() } )
oThread:Run()
```
"""
    ),
    create_example(
        "Implementar mutex para proteger variables compartidas",
        """Mutex para sincronización:

```harbour
CLASS ThreadSafeCounter
   DATA nCount AS 0
   DATA oMutex

   METHOD New()
   METHOD Increment()
   METHOD Decrement()
   METHOD Get()
ENDCLASS

METHOD New() CLASS ThreadSafeCounter
   ::oMutex := TMutex():New()
   RETURN SELF

METHOD Increment() CLASS ThreadSafeCounter
   ::oMutex:Lock()
   ::nCount++
   ::oMutex:UnLock()
   RETURN ::nCount

METHOD Decrement() CLASS ThreadSafeCounter
   ::oMutex:Lock()
   ::nCount--
   ::oMutex:UnLock()
   RETURN ::nCount

METHOD Get() CLASS ThreadSafeCounter
   RETURN ::nCount
```

IMPORTANTE: Siempre usar Lock/UnLock para variables compartidas.
"""
    ),
    create_example(
        "Crear un pool de hilos para procesamiento paralelo",
        """Pool de hilos eficiente:

```harbour
CLASS ThreadPool
   DATA aThreads
   DATA nMaxThreads
   DATA oQueue
   DATA oMutex

   METHOD New( nMax )
   METHOD Submit( bTask )
   METHOD ProcessNext()
   METHOD Shutdown()
ENDCLASS

METHOD New( nMax ) CLASS ThreadPool
   DEFAULT nMax TO 4
   ::nMaxThreads := nMax
   ::aThreads := {}
   ::oQueue := ArrayList():New()
   ::oMutex := TMutex():New()
   RETURN SELF

METHOD Submit( bTask ) CLASS ThreadPool
   ::oMutex:Lock()
   ::oQueue:Add( bTask )
   ::oMutex:UnLock()
   ::ProcessNext()
   RETURN NIL

METHOD ProcessNext() CLASS ThreadPool
   LOCAL bTask, oThread
   IF Len( ::aThreads ) < ::nMaxThreads .AND. ::oQueue:Size() > 0
      ::oMutex:Lock()
      bTask := ::oQueue:RemoveAt( 1 )
      ::oMutex:UnLock()
      oThread := Thread():New( bTask )
      AAdd( ::aThreads, oThread )
      oThread:Run()
   ENDIF
   RETURN NIL

METHOD Shutdown() CLASS ThreadPool
   LOCAL oThread
   FOR EACH oThread IN ::aThreads
      oThread:Join()
   NEXT
   RETURN NIL
```
"""
    ),
    create_example(
        "¿Cómo usar eventos entre hilos?",
        """Comunicación entre hilos con mensajes:

```harbour
CLASS InterThreadMessenger
   DATA oQueue
   DATA oMutex
   DATA hWnd

   METHOD New( hWnd )
   METHOD Send( uMessage )
   METHOD Receive()
ENDCLASS

METHOD New( hWnd ) CLASS InterThreadMessenger
   ::hWnd := hWnd
   ::oQueue := ArrayList():New()
   ::oMutex := TMutex():New()
   RETURN SELF

METHOD Send( uMessage ) CLASS InterThreadMessenger
   ::oMutex:Lock()
   ::oQueue:Add( uMessage )
   ::oMutex:UnLock()
   PostMessage( ::hWnd, WM_USER + 1, 0, 0 )
   RETURN NIL

METHOD Receive() CLASS InterThreadMessenger
   LOCAL uMessage := NIL
   ::oMutex:Lock()
   IF ::oQueue:Size() > 0
      uMessage := ::oQueue:RemoveAt( 1 )
   ENDIF
   ::oMutex:UnLock()
   RETURN uMessage
```

En el hilo principal, manejar WM_USER + 1 para recibir mensajes.
"""
    ),
    create_example(
        "Crear un semáforo para limitar concurrencia",
        """Semáforo para control de acceso:

```harbour
CLASS Semaphore
   DATA nMax
   DATA nCurrent AS 0
   DATA oMutex
   DATA oCondition

   METHOD New( nMax )
   METHOD Wait()
   METHOD Signal()
   METHOD TryWait()
ENDCLASS

METHOD New( nMax ) CLASS Semaphore
   ::nMax := nMax
   ::oMutex := TMutex():New()
   ::oCondition := Condition():New( ::oMutex )
   RETURN SELF

METHOD Wait() CLASS Semaphore
   ::oMutex:Lock()
   WHILE ::nCurrent >= ::nMax
      ::oCondition:Wait()
   ENDDO
   ::nCurrent++
   ::oMutex:UnLock()
   RETURN NIL

METHOD Signal() CLASS Semaphore
   ::oMutex:Lock()
   ::nCurrent--
   ::oCondition:Signal()
   ::oMutex:UnLock()
   RETURN NIL

METHOD TryWait() CLASS Semaphore
   LOCAL lAcquired := .F.
   ::oMutex:Lock()
   IF ::nCurrent < ::nMax
      ::nCurrent++
      lAcquired := .T.
   ENDIF
   ::oMutex:UnLock()
   RETURN lAcquired
```

Uso:
```harbour
oSem := Semaphore():New( 3 )  // Max 3 hilos concurrentes
```
"""
    ),
]

# Networking examples
NETWORKING = [
    create_example(
        "Crear un servidor TCP básico en Harbour",
        """Servidor TCP con FiveWin:

```harbour
#include "fivewin.ch"

CLASS TCPServer
   DATA nPort
   DATA oServerSocket
   DATA aClients AS {}
   DATA bOnData
   DATA bOnConnect

   METHOD New( nPort )
   METHOD Start()
   METHOD Stop()
   METHOD OnAccept( oSocket )
   METHOD Broadcast( cMessage )
ENDCLASS

METHOD New( nPort ) CLASS TCPServer
   ::nPort := nPort
   RETURN SELF

METHOD Start() CLASS TCPServer
   ::oServerSocket := TServerSocket():New( ::nPort )
   ::oServerSocket:bOnAccept := {|o| ::OnAccept(o) }
   ::oServerSocket:Listen()
   RETURN NIL

METHOD OnAccept( oSocket ) CLASS TCPServer
   LOCAL oClient := oSocket:Accept()
   AAdd( ::aClients, oClient )
   oClient:bOnData := {|o,c| ::bOnData(o,c) }
   IF ::bOnConnect != NIL
      EVAL( ::bOnConnect, oClient )
   ENDIF
   RETURN NIL

METHOD Broadcast( cMessage ) CLASS TCPServer
   LOCAL oClient
   FOR EACH oClient IN ::aClients
      oClient:Send( cMessage )
   NEXT
   RETURN NIL

METHOD Stop() CLASS TCPServer
   LOCAL oClient
   FOR EACH oClient IN ::aClients
      oClient:Close()
   NEXT
   ::oServerSocket:Close()
   RETURN NIL
```
"""
    ),
    create_example(
        "Hacer una petición HTTP GET con Harbour",
        """GET HTTP con THttpClient:

```harbour
FUNCTION HttpGet( cUrl, aHeaders )
   LOCAL oHttp, cResponse, aHeader
   
   oHttp := THttpClient():New()
   oHttp:SetUrl( cUrl )
   oHttp:SetTimeout( 30 )
   
   IF PCount() > 1 .AND. ValType( aHeaders ) == "A"
      FOR EACH aHeader IN aHeaders
         oHttp:AddHeader( aHeader[1], aHeader[2] )
      NEXT
   ENDIF
   
   IF oHttp:Execute( "GET" )
      cResponse := oHttp:GetResponse()
   ELSE
      cResponse := '{"error": "' + oHttp:GetError() + '"}'
   ENDIF
   
   oHttp:Close()
   RETURN cResponse
```

Uso:
```harbour
cJson := HttpGet( "https://api.example.com/users" )
aData := HB_JSonDecode( cJson )
```
"""
    ),
    create_example(
        "Enviar datos con POST HTTP en Harbour",
        """POST HTTP con datos JSON:

```harbour
FUNCTION HttpPost( cUrl, cJsonData, aHeaders )
   LOCAL oHttp, cResponse
   
   oHttp := THttpClient():New()
   oHttp:SetUrl( cUrl )
   oHttp:SetTimeout( 30 )
   oHttp:AddHeader( "Content-Type", "application/json" )
   
   IF PCount() > 2 .AND. ValType( aHeaders ) == "A"
      LOCAL aHeader
      FOR EACH aHeader IN aHeaders
         oHttp:AddHeader( aHeader[1], aHeader[2] )
      NEXT
   ENDIF
   
   IF oHttp:Execute( "POST", cJsonData )
      cResponse := oHttp:GetResponse()
   ELSE
      cResponse := '{"error": "' + oHttp:GetError() + '"}'
   ENDIF
   
   oHttp:Close()
   RETURN cResponse
```

Uso:
```harbour
cData := '{"name": "Juan", "email": "juan@test.com"}'
cResult := HttpPost( "https://api.example.com/users", cData )
```
"""
    ),
    create_example(
        "Implementar cliente WebSocket en Harbour",
        """WebSocket client:

```harbour
CLASS WebSocketClient
   DATA oSocket
   DATA cUrl
   DATA lConnected AS .F.
   DATA bOnOpen
   DATA bOnMessage
   DATA bOnClose
   DATA bOnError

   METHOD New( cUrl )
   METHOD Connect()
   METHOD Send( cMessage )
   METHOD Close()
   METHOD OnOpen()
   METHOD OnMessage( cData )
   METHOD OnClose()
ENDCLASS

METHOD New( cUrl ) CLASS WebSocketClient
   ::cUrl := cUrl
   RETURN SELF

METHOD Connect() CLASS WebSocketClient
   ::oSocket := TWebSocket():New( ::cUrl )
   ::oSocket:bOnOpen := {|o| ::OnOpen() }
   ::oSocket:bOnMessage := {|o,c| ::OnMessage(c) }
   ::oSocket:bOnClose := {|o| ::OnClose() }
   ::oSocket:Connect()
   ::lConnected := .T.
   RETURN NIL

METHOD Send( cMessage ) CLASS WebSocketClient
   IF ::lConnected
      ::oSocket:Send( cMessage )
   ENDIF
   RETURN NIL

METHOD Close() CLASS WebSocketClient
   IF ::lConnected
      ::oSocket:Close()
      ::lConnected := .F.
   ENDIF
   RETURN NIL

METHOD OnOpen() CLASS WebSocketClient
   ::lConnected := .T.
   IF ::bOnOpen != NIL
      EVAL( ::bOnOpen )
   ENDIF
   RETURN NIL

METHOD OnMessage( cData ) CLASS WebSocketClient
   IF ::bOnMessage != NIL
      EVAL( ::bOnMessage, cData )
   ENDIF
   RETURN NIL

METHOD OnClose() CLASS WebSocketClient
   ::lConnected := .F.
   IF ::bOnClose != NIL
      EVAL( ::bOnClose )
   ENDIF
   RETURN NIL
```
"""
    ),
    create_example(
        "Crear un cliente FTP para subir archivos",
        """FTP client con FiveWin:

```harbour
CLASS FTPClient
   DATA oFTP
   DATA cServer
   DATA cUser
   DATA cPass

   METHOD New( cServer, cUser, cPass )
   METHOD Connect()
   METHOD Upload( cLocalFile, cRemoteFile )
   METHOD Download( cRemoteFile, cLocalFile )
   METHOD Disconnect()
ENDCLASS

METHOD New( cServer, cUser, cPass ) CLASS FTPClient
   ::cServer := cServer
   ::cUser := cUser
   ::cPass := cPass
   RETURN SELF

METHOD Connect() CLASS FTPClient
   ::oFTP := TFtpClient():New()
   ::oFTP:SetServer( ::cServer )
   ::oFTP:SetUser( ::cUser )
   ::oFTP:SetPassword( ::cPass )
   RETURN ::oFTP:Connect()

METHOD Upload( cLocalFile, cRemoteFile ) CLASS FTPClient
   DEFAULT cRemoteFile TO FileBaseName( cLocalFile )
   RETURN ::oFTP:PutFile( cLocalFile, cRemoteFile )

METHOD Download( cRemoteFile, cLocalFile ) CLASS FTPClient
   DEFAULT cLocalFile TO FileBaseName( cRemoteFile )
   RETURN ::oFTP:GetFile( cRemoteFile, cLocalFile )

METHOD Disconnect() CLASS FTPClient
   IF ::oFTP != NIL
      ::oFTP:Disconnect()
   ENDIF
   RETURN NIL
```
"""
    ),
]

# Error handling examples
ERROR_HANDLING = [
    create_example(
        "Explica BEGIN SEQUENCE y RECOVER en Harbour",
        """BEGIN SEQUENCE es el mecanismo structured de error handling:

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

Con THROW para errores personalizados:
```harbour
BEGIN SEQUENCE
   IF Empty( cData )
      THROW( ErrorNew( "MYAPP", 0, "Datos inválidos" ) )
   ENDIF
   ProcessData( cData )
RECOVER USING oError
   HandleError( oError )
END SEQUENCE
```
"""
    ),
    create_example(
        "Crear sistema de logging de errores",
        """Logger profesional:

```harbour
CLASS ErrorLogger
   DATA cLogFile
   DATA oMutex
   DATA cLevel AS "INFO"

   METHOD New( cLogFile )
   METHOD Log( cMessage, cLevel )
   METHOD Error( cMessage )
   METHOD Warn( cMessage )
   METHOD Info( cMessage )
   METHOD GetTimestamp()
ENDCLASS

METHOD New( cLogFile ) CLASS ErrorLogger
   ::cLogFile := cLogFile
   ::oMutex := TMutex():New()
   RETURN SELF

METHOD Log( cMessage, cLevel ) CLASS ErrorLogger
   LOCAL cLine
   DEFAULT cLevel TO "INFO"
   cLine := ::GetTimestamp() + " [" + cLevel + "] " + cMessage
   ::oMutex:Lock()
   HB_MemoWrit( ::cLogFile, cLine + HB_EOL(), .T. )
   ::oMutex:UnLock()
   RETURN NIL

METHOD Error( cMessage ) CLASS ErrorLogger
   RETURN ::Log( cMessage, "ERROR" )

METHOD Warn( cMessage ) CLASS ErrorLogger
   RETURN ::Log( cMessage, "WARN" )

METHOD Info( cMessage ) CLASS ErrorLogger
   RETURN ::Log( cMessage, "INFO" )

METHOD GetTimestamp() CLASS ErrorLogger
   RETURN DTOS( Date() ) + " " + Time()
```
"""
    ),
    create_example(
        "Implementar retry con backoff exponencial",
        """Retry pattern:

```harbour
FUNCTION RetryWithBackoff( bOperation, nMaxRetries, nBaseDelay, nMaxDelay )
   LOCAL nAttempt := 0, nDelay, lSuccess := .F., oError
   
   DEFAULT nMaxRetries TO 3
   DEFAULT nBaseDelay TO 1000
   DEFAULT nMaxDelay TO 30000
   
   WHILE !lSuccess .AND. nAttempt < nMaxRetries
      BEGIN SEQUENCE
         EVAL( bOperation )
         lSuccess := .T.
      RECOVER USING oError
         nAttempt++
         nDelay := Min( nBaseDelay * (2 ^ (nAttempt - 1)), nMaxDelay )
         WaitSeconds( nDelay / 1000 )
      END SEQUENCE
   ENDDO
   
   IF !lSuccess
      THROW( oError )
   ENDIF
   
   RETURN NIL

FUNCTION WaitSeconds( nSeconds )
   LOCAL nStart := Seconds()
   WHILE Seconds() - nStart < nSeconds
      // Busy wait
   ENDDO
   RETURN NIL
```
"""
    ),
    create_example(
        "Manejar errores de red con reconexión automática",
        """Reconexión automática:

```harbour
CLASS ResilientConnection
   DATA oConnection
   DATA cServer
   DATA nPort
   DATA nMaxRetries
   DATA nRetryDelay
   DATA lConnected AS .F.

   METHOD New( cServer, nPort )
   METHOD Connect()
   METHOD Reconnect()
   METHOD Send( cData )
   METHOD OnError( oError )
ENDCLASS

METHOD New( cServer, nPort ) CLASS ResilientConnection
   ::cServer := cServer
   ::nPort := nPort
   ::nMaxRetries := 5
   ::nRetryDelay := 1000
   RETURN SELF

METHOD Connect() CLASS ResilientConnection
   LOCAL nAttempt := 0
   WHILE !::lConnected .AND. nAttempt < ::nMaxRetries
      BEGIN SEQUENCE
         ::oConnection := TSocket():New( ::cServer, ::nPort )
         ::oConnection:Connect()
         ::lConnected := .T.
      RECOVER USING oError
         nAttempt++
         Sleep( ::nRetryDelay * nAttempt )
      END SEQUENCE
   ENDDO
   RETURN ::lConnected

METHOD Reconnect() CLASS ResilientConnection
   ::lConnected := .F.
   BEGIN SEQUENCE
      IF ::oConnection != NIL
         ::oConnection:Close()
      ENDIF
   RECOVER
   END SEQUENCE
   RETURN ::Connect()

METHOD Send( cData ) CLASS ResilientConnection
   IF !::lConnected
      ::Reconnect()
   ENDIF
   IF ::lConnected
      BEGIN SEQUENCE
         ::oConnection:Send( cData )
      RECOVER USING oError
         ::Reconnect()
         ::oConnection:Send( cData )
      END SEQUENCE
   ENDIF
   RETURN NIL
```
"""
    ),
]

# Database examples
DATABASE = [
    create_example(
        "Conectar y consultar un archivo DBF en Harbour",
        """Operaciones DBF básicas:

```harbour
FUNCTION ConsultaDBF( cDbfPath )
   LOCAL nArea := select()
   
   USE ( cDbfPath ) ALIAS "miarchivo" SHARED READONLY
   
   IF neterr()
      ? "Error al abrir archivo"
      RETURN NIL
   ENDIF
   
   GO TOP
   WHILE !EOF()
      ? FIELDGET(1), FIELDGET(2), FIELDGET(3)
      SKIP
   ENDDO
   
   USE
   SELECT ( nArea )
   RETURN NIL
```

Con filtro:
```harbour
USE clientes SHARED
SET FILTER TO estado == "ACTIVO"
GO TOP
```
"""
    ),
    create_example(
        "Realizar transacciones con Firebird/PostgreSQL",
        """SQL transacciones:

```harbour
FUNCTION SQLTransaction( cServer, cDB, cUser, cPass )
   LOCAL oDB, lSuccess := .F.
   
   oDB := TSQLConnection():New( ;
      "PGSQL", cServer, cDB, cUser, cPass )
   
   IF oDB:Connect()
      oDB:BeginTransaction()
      
      BEGIN SEQUENCE
         oDB:Execute( ;
            "INSERT INTO ventas (fecha, total) VALUES (NOW(), 1000)" )
         oDB:Execute( ;
            "UPDATE stock SET cantidad = cantidad - 1 WHERE prod_id = 5" )
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
    ),
    create_example(
        "Crear un ORM simple para Harbour",
        """ORM básico:

```harbour
CLASS Model
   DATA cTable
   DATA aFields AS {}
   DATA oDB

   METHOD New( cTable, oDB )
   METHOD Find( nId )
   METHOD Save()
   METHOD Delete()
   METHOD ToJson()
   METHOD FromJson( cJson )
ENDCLASS

METHOD New( cTable, oDB ) CLASS Model
   ::cTable := cTable
   ::oDB := oDB
   RETURN SELF

METHOD Find( nId ) CLASS Model
   LOCAL cSQL := "SELECT * FROM " + ::cTable + " WHERE id = " + AllTrim( Str( nId ) )
   RETURN ::oDB:Query( cSQL )

METHOD Save() CLASS Model
   LOCAL cSQL, aValues := {}, aFields := {}, nI
   FOR nI := 1 TO Len( ::aFields )
      IF ::aFields[nI][2] != NIL
         AAdd( aFields, ::aFields[nI][1] )
         AAdd( aValues, ::aFields[nI][2] )
      ENDIF
   NEXT
   cSQL := "INSERT INTO " + ::cTable + ;
           " (" + HB_ArrayToCSV( aFields ) + ")" + ;
           " VALUES (" + HB_ArrayToCSV( aValues ) + ")"
   RETURN ::oDB:Execute( cSQL )

METHOD Delete() CLASS Model
   LOCAL cSQL := "DELETE FROM " + ::cTable + " WHERE id = " + ;
                 AllTrim( Str( ::nId ) )
   RETURN ::oDB:Execute( cSQL )

METHOD ToJson() CLASS Model
   LOCAL oJson := HB_JsonObject():New()
   LOCAL aField
   FOR EACH aField IN ::aFields
      oJson[ aField[1] ] := aField[2]
   NEXT
   RETURN HB_JSonEncode( oJson )
```
"""
    ),
    create_example(
        "Implementar caché de consulta de base de datos",
        """Caché de queries:

```harbour
CLASS QueryCache
   DATA hCache AS {=>}
   DATA oDB
   DATA nTTL AS 300  // 5 minutos

   METHOD New( oDB )
   METHOD Query( cSQL )
   METHOD Invalidate( cKey )
   METHOD Clear()
   METHOD IsExpired( cKey )
ENDCLASS

METHOD New( oDB ) CLASS QueryCache
   ::oDB := oDB
   RETURN SELF

METHOD Query( cSQL ) CLASS QueryCache
   LOCAL cKey := HB_M5( cSQL )
   IF ::IsExpired( cKey )
      ::hCache[ cKey ] := { ;
         "data" => ::oDB:Query( cSQL ), ;
         "time" => Seconds() }
   ENDIF
   RETURN ::hCache[ cKey ][ "data" ]

METHOD IsExpired( cKey ) CLASS QueryCache
   IF !HB_HHasKey( ::hCache, cKey )
      RETURN .T.
   ENDIF
   RETURN ( Seconds() - ::hCache[ cKey ][ "time" ] ) > ::nTTL

METHOD Invalidate( cKey ) CLASS QueryCache
   IF HB_HHasKey( ::hCache, cKey )
      HB_HDel( ::hCache, cKey )
   ENDIF
   RETURN NIL

METHOD Clear() CLASS QueryCache
   ::hCache := {=>}
   RETURN NIL
```
"""
    ),
]

# OOP examples
OOP = [
    create_example(
        "Explica la herencia en Harbour/OOP",
        """Herencia en Harbour:

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
   RETURN "¡Guau!"

METHOD Fetch() CLASS Dog
   RETURN ::cName + " trae la pelota"
```

Uso:
```harbour
oDog := Dog():New("Max", 3, "Labrador")
? oDog:Speak()    // ¡Guau!
? oDog:Describe()  // Max tiene 3 años
```
"""
    ),
    create_example(
        "Implementar patrón Singleton en Harbour",
        """Singleton pattern:

```harbour
CLASS Singleton
   CLASSDATA oInstance AS NIL

   METHOD New()
   CLASS METHOD GetInstance()
ENDCLASS

METHOD New() CLASS Singleton
   IF ::oInstance != NIL
      THROW( ErrorNew( "SINGLETON", 0, "Use GetInstance()" ) )
   ENDIF
   ::oInstance := SELF
   RETURN SELF

CLASS METHOD GetInstance() CLASS Singleton
   IF ::oInstance == NIL
      ::oInstance := Singleton():New()
   ENDIF
   RETURN ::oInstance
```

Uso:
```harbour
oSingleton := Singleton():GetInstance()
```
"""
    ),
    create_example(
        "Crear una interfaz (mixin) en Harbour",
        """Patrón mixin:

```harbour
// Interfaz de serialización
CLASS Serializable
   METHOD Serialize()
   METHOD Deserialize( cData )
ENDCLASS

METHOD Serialize() CLASS Serializable
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

Uso:
```harbour
oUser := User():New("Juan", "juan@test.com")
cJson := oUser:Serialize()
```
"""
    ),
    create_example(
        "Implementar patrón Observer en Harbour",
        """Observer pattern:

```harbour
CLASS Observable
   DATA aObservers AS {}

   METHOD AddObserver( oObserver )
   METHOD RemoveObserver( oObserver )
   METHOD Notify( cEvent, uData )
ENDCLASS

METHOD AddObserver( oObserver ) CLASS Observable
   AAdd( ::aObservers, oObserver )
   RETURN NIL

METHOD RemoveObserver( oObserver ) CLASS Observable
   LOCAL nPos := AScan( ::aObservers, oObserver )
   IF nPos > 0
      ADel( ::aObservers, nPos )
      ASize( ::aObservers, Len( ::aObservers ) - 1 )
   ENDIF
   RETURN NIL

METHOD Notify( cEvent, uData ) CLASS Observable
   LOCAL oObserver
   FOR EACH oObserver IN ::aObservers
      oObserver:Update( cEvent, uData )
   NEXT
   RETURN NIL

// Observer interface
CLASS Observer
   METHOD Update( cEvent, uData )
ENDCLASS

METHOD Update( cEvent, uData ) CLASS Observer
   // Override in subclasses
   RETURN NIL
```
"""
    ),
    create_example(
        "Crear un Factory en Harbour",
        """Factory pattern:

```harbour
CLASS ShapeFactory
   CLASS METHOD Create( cType, ... )

   CLASS METHOD Register( cType, bCreator )
   CLASS DATA hCreators AS {=>}
ENDCLASS

CLASS METHOD Create( cType, ... ) CLASS ShapeFactory
   IF HB_HHasKey( ::hCreators, cType )
      RETURN EVAL( ::hCreators[ cType ], ... )
   ENDIF
   THROW( ErrorNew( "FACTORY", 0, "Tipo desconocido: " + cType ) )
   RETURN NIL

CLASS METHOD Register( cType, bCreator ) CLASS ShapeFactory
   ::hCreators[ cType ] := bCreator
   RETURN NIL

// Uso
CLASS Circle
   DATA nRadius
   METHOD New( nRadius )
ENDCLASS

METHOD New( nRadius ) CLASS Circle
   ::nRadius := nRadius
   RETURN SELF

// Registrar
ShapeFactory:Register( "circle", {|r| Circle():New(r) } )

// Crear
oShape := ShapeFactory:Create( "circle", 5 )
```
"""
    ),
]

# Fix/Debug examples
FIX_DEBUG = [
    create_example(
        "Mi código DBF no cierra bien, ¿qué hago mal?",
        """Problema: no se verifica antes de cerrar.

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
   SELECT 1
   USE
ENDIF
IF Used(2)
   SELECT 2
   USE
ENDIF
```
"""
    ),
    create_example(
        "SET FILTER TO no funciona, ¿por qué?",
        """Causas comunes:

1. Falta macro:
```harbour
SET FILTER TO estado = "A"  // MAL
SET FILTER TO estado == "A"  // BIEN
```

2. No hay GO TOP:
```harbour
SET FILTER TO estado = "A"
GO TOP  // ¡Imprescindible!
```

3. Usar DbSetFilter():
```harbour
DbSetFilter( {|| estado == "A" }, 'estado == "A"' )
GO TOP
```
"""
    ),
    create_example(
        "El programa se cuelga al acceder a BD, ¿cómo lo soluciono?",
        """Causas comunes de bloqueo:

1. Archivo no compartido:
```harbour
USE archivo SHARED  // ¡Agregar SHARED!
```

2. Sin manejo de errores:
```harbour
BEGIN SEQUENCE
   USE archivo SHARED
   IF neterr()
      THROW( ErrorNew( "DBF", 0, "No se pudo abrir" ) )
   ENDIF
RECOVER USING oError
   ? "Error:", oError:description
END SEQUENCE
```

3. Timeout de red:
```harbour
SET HARDLOCK TO 5  // 5 segundos timeout
```
"""
    ),
    create_example(
        "Cómo optimizar una consulta lenta en Harbour",
        """Optimización de consultas:

1. Usar índices:
```harbour
USE clientes
INDEX ON codigo TO clientes.idx
SEEK "001"
```

2. Filtros eficientes:
```harbour
SET FILTER TO estado == "A"  // Comparación exacta
```

3. Evitar SEEK en bucle:
```harbour
// MAL
GO TOP
WHILE !EOF()
   SEEK FIELDGET(1)
   SKIP
ENDDO

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
```
"""
    ),
]

def main():
    print("🚀 Generando dataset mejorado Harbour/FWH...")
    
    # Load existing
    examples = []
    with open('/home/antonio/finetune/harbour_fwh_dedup.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"📥 Cargados {len(examples)} ejemplos existentes")
    
    # Add new examples
    new_examples = MULTITHREADING + NETWORKING + ERROR_HANDLING + DATABASE + OOP + FIX_DEBUG
    print(f"📝 Generados {len(new_examples)} nuevos ejemplos")
    
    # Combine
    all_examples = examples + new_examples
    
    # Deduplicate
    seen = set()
    unique = []
    for ex in all_examples:
        key = ex['instruction']
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    
    print(f"📊 Total único: {len(unique)}")
    
    # Save
    output_path = '/home/antonio/finetune/harbour_fwh_improved.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in unique:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✅ Guardado en: {output_path}")
    print(f"📈 Mejora: +{len(unique) - len(examples)} ejemplos")

if __name__ == "__main__":
    main()
