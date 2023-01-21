package com.sachin.objectprediction

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.sachin.objectprediction.ml.Iris
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val ed1 = findViewById<EditText>(R.id.editTextNumberDecimal)
        val ed2 = findViewById<EditText>(R.id.editTextNumberDecimal2)
        val ed3 = findViewById<EditText>(R.id.editTextNumberDecimal3)
        val ed4 = findViewById<EditText>(R.id.editTextNumberDecimal4)
        findViewById<Button>(R.id.button).setOnClickListener {
            val v1 = ed1.text.toString().toFloat()
            val v2 = ed2.text.toString().toFloat()
            val v3 = ed3.text.toString().toFloat()
            val v4 = ed4.text.toString().toFloat()

            val byteBuffer = ByteBuffer.allocateDirect(4 * 4)
            byteBuffer.putFloat(v1)
            byteBuffer.putFloat(v2)
            byteBuffer.putFloat(v3)
            byteBuffer.putFloat(v4)

            val model = Iris.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            val tv = findViewById<TextView>(R.id.textView)
            tv.text = "Iris-setosa = "+outputFeature0[0].toString()
                .plus("\n").plus("Iris-versicolor = ${outputFeature0[1].toString()}"
                    .plus("\n").plus("Iris-virginica = ${outputFeature0[2].toString()}"))

            // Releases model resources if no longer used.
            model.close()
        }
    }
}