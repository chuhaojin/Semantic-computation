<?php
header("Content-Type:text/html;charset=UTF-8");
include "class_word2vec.php";
$deal=new word2vec();
$result=$deal->get_similarity($_GET);
echo json_encode($result);
/**11
 * Created by PhpStorm.
 * User: vortex
 * Date: 2018/1/27
 * Time: 10:09
 */