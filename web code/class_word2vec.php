<?php
error_reporting(E_ALL);
set_time_limit(0);
header("Content-Type:text/html;charset=utf-8");
function BinToStr($str){
    $arr = explode(' ', $str);
    foreach($arr as &$v){
        $v = pack("H".strlen(base_convert($v, 2, 16)), base_convert($v, 2, 16));
    }

    return join('', $arr);
}
class word2vec{
//    public function get_similarity($word)
//    {
//        $file = fopen("word.txt","w")or die("Unable to open word.txt!");
//        fwrite($file, $word['word1']."##".$word['word2']);
//        fclose($file);
//        $time1=microtime();
//        while(!file_exists("smlrt.txt")){
//            $time2=microtime();
//            if($time2-$time1>2000000){
//                return false;
//            }
//            usleep(10000);
//        }
//        $file = fopen("smlrt.txt","r")or die("Unable to open smlrt.txt!");
//        $smlrt=fgets($file);
//        fclose($file);
//        unlink('smlrt.txt');
//        $data=array('word1'=>$word['word1'],'word2'=>$word['word2'],'smlrt'=>$smlrt);
//        return $data;
//    }

    public function get_similarity($word)
    {
        $port=10271;
        $ip="127.0.0.1";
        $socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
        $result = socket_connect($socket, $ip, $port);
//        if ($result < 0)
//        {
//            echo "socket_connect() failed.\nReason: ($result) " . socket_strerror($result) . "\n";
//        }
        if(!socket_write($socket, "1##".$word["word1"]."##".$word["word2"], 3+strlen($word["word1"]."##".$word["word2"])))
        {
//            echo "socket_write() failed: reason: " . socket_strerror($socket) . "\n";
        }
        $last = "-1";
        while($out = socket_read($socket, 8192)) {
            $last = $out;
        }
        socket_close($socket);
        $data=array('word1'=>$word['word1'],'word2'=>$word['word2'],'last'=>$last);
//        echo "out:".$last."\n";
//        echo gettype($last);
        return $data;
    }
    public function get_word_operation($word)
    {
        $port=10271;
        $ip="127.0.0.1";
        $socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
        $result = socket_connect($socket, $ip, $port);
//        if ($result < 0)
//        {
//            echo "socket_connect() failed.\nReason: ($result) " . socket_strerror($result) . "\n";
//        }
        if(!socket_write($socket, "2##".$word["sentence"], 3+strlen($word["sentence"])))
        {
//            echo "socket_write() failed: reason: " . socket_strerror($socket) . "\n";
        }
        $last = "-1";
        while($out = socket_read($socket, 8192)) {
            $last = $out;
        }
        socket_close($socket);
        $data=array('sentence'=>$word['sentence'],'result'=>$last);
//        echo "out:".$last."\n";
//        echo gettype($last);
        return $data;
    }
}
