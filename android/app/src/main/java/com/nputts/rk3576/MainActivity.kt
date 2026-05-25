package com.nputts.rk3576

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.*

class MainActivity : ComponentActivity() {
    private val engine = NpuTtsEngine()
    private val player = AudioPlayer()
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        player.init()

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    TtsScreen(
                        onSynthesize = { text ->
                            synthesizeAndPlay(text)
                        },
                        isSpeaking = engine.isSpeaking(),
                        isReady = engine.isReady()
                    )
                }
            }
        }

        // Auto-init engine
        scope.launch(Dispatchers.IO) {
            val modelDir = "/sdcard/npu-tts-models"
            val ok = engine.init(modelDir)
            withContext(Dispatchers.Main) {
                if (ok) {
                    Toast.makeText(this@MainActivity, "NPU TTS ready", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Init failed. Push models to $modelDir/",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private fun synthesizeAndPlay(text: String) {
        scope.launch(Dispatchers.IO) {
            val pcm = engine.synthesize(text)
            if (pcm != null) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity,
                        "Playing ${pcm.size / NpuTtsEngine.SAMPLE_RATE.toFloat().format(1)}s audio",
                        Toast.LENGTH_SHORT).show()
                }
                player.play(pcm)
            } else {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Synthesis failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun Float.format(digits: Int) = String.format("%.${digits}f", this)

    override fun onDestroy() {
        scope.cancel()
        engine.release()
        player.release()
        super.onDestroy()
    }
}

@Composable
fun TtsScreen(
    onSynthesize: (String) -> Unit,
    isSpeaking: Boolean,
    isReady: Boolean
) {
    var text by remember { mutableStateOf("你好世界，这是一个NPU语音合成测试。") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "NPU TTS RK3576",
            style = MaterialTheme.typography.headlineMedium
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = if (isReady) "Engine: Ready (CPU encoder + NPU decoder)" else "Engine: Loading...",
            style = MaterialTheme.typography.bodyMedium,
            color = if (isReady) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
        )

        Spacer(modifier = Modifier.height(24.dp))

        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("输入中文文本") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3,
            maxLines = 6,
            enabled = isReady && !isSpeaking
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { onSynthesize(text) },
            enabled = isReady && !isSpeaking && text.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isSpeaking) "Speaking..." else "Synthesize (NPU)")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Models: /sdcard/npu-tts-models/\n" +
                    "  encoder.onnx (CPU)\n" +
                    "  decoder_rk3576.rknn (NPU)\n" +
                    "  g.bin (speaker embedding)",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}
