<?php
header("Content-Type:text/html;charset=UTF-8");
include "class_word2vec.php";
$deal=new word2vec();
$result=$deal->get_word_operation($_GET);
echo json_encode($result);
/**
 * Created by PhpStorm.
 * User: vortex
 * Date: 2018/9/23
 * Time: 14:28
 */