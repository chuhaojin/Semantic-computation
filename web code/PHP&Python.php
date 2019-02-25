<?php
$order = 'python D:\hello.py';
$file = fopen("word.txt","w")or die("Unable to open file!");
$txt = "Bill Gates\n";
fwrite($file, $txt);
fclose($file);
/**
 * Created by PhpStorm.
 * User: vortex
 * Date: 2018/1/27
 * Time: 15:15
 */