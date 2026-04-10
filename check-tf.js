async function check() {
    try {
        console.log('Загрузка @tensorflow/tfjs-node-gpu...');
        const tf = require('@tensorflow/tfjs-node-gpu');
        await tf.setBackend('tensorflow');
        await tf.ready();
        console.log('Пакет успешно загружен.');

        console.log('Доступные бэкенды:', Object.keys(tf.engine().registry));
        console.log('Текущий бэкенд:', tf.getBackend());
        const backend = tf.engine().backendInstance;
        const usingGpu = typeof backend?.isUsingGpuDevice === 'function'
            ? backend.isUsingGpuDevice()
            : backend?.binding && typeof backend.binding.isUsingGpuDevice === 'function'
                ? backend.binding.isUsingGpuDevice()
                : Boolean(backend?.isUsingGpuDevice);
        console.log('Используется GPU:', usingGpu);

        if (tf.getBackend() === 'tensorflow') {
            console.log('Бэкенд "tensorflow" активен. Попытка выполнить контрольную операцию...');
            const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
            const b = tf.tensor2d([5, 6, 7, 8], [2, 2]);
            const c = a.matMul(b);

            console.log('Результат операции:');
            c.print();
            a.dispose();
            b.dispose();
            c.dispose();

            if (usingGpu) {
                console.log('\n\x1b[32m%s\x1b[0m', 'УСПЕХ: TensorFlow.js действительно использует GPU.');
            } else {
                console.log('\n\x1b[33m%s\x1b[0m', 'ВНИМАНИЕ: backend tensorflow активен, но вычисления идут без GPU.');
            }
        } else {
            console.log('\n\x1b[31m%s\x1b[0m', 'ОШИБКА: Бэкенд "tensorflow" (GPU) не был активирован.');
        }
    } catch (e) {
        console.error('\n\x1b[31m%s\x1b[0m', 'Произошла критическая ошибка при инициализации TensorFlow:');
        console.error(e);
    }
}

check();
