package com.a303.consultms.domain.channel.controller;

import com.a303.consultms.domain.channel.Channel;
import com.a303.consultms.domain.channel.dto.response.ChannelListRes;
import com.a303.consultms.domain.channel.dto.response.ChannelRes;
import com.a303.consultms.domain.channel.service.ChannelService;
import com.a303.consultms.domain.consult.service.ConsultService;
import com.a303.consultms.domain.message.Message;
import com.a303.consultms.global.api.response.BaseResponse;
import com.a303.consultms.global.exception.code.SuccessCode;
import jakarta.transaction.Transactional;
import java.io.IOException;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import zipkin2.Call.Base;

@RestController
@RequestMapping("/api/channels/v1")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ChannelController {

    private final ChannelService channelService;
    private final ConsultService consultService;

    //고민상담소 입장(채팅방 생성)
    @PostMapping("/enter/{consultId}")
    @Transactional
    public ResponseEntity<BaseResponse<String>> registerPersonalChat(
        @PathVariable int consultId,
        @RequestHeader("x-member-id") int sender //입장하는 사람
    ) throws IOException {

        //고민상담소 입장
        Channel channel = consultService.registerChannel(consultId, sender);

        // TODO 동준이가 나중에 해결
//        consultService.makeNotification("ENTER", Integer.parseInt(channel.getReceiver().get("memberId")));

        return BaseResponse.success(SuccessCode.INSERT_SUCCESS, channel.getChannelId());
    }

    //참여중인 채팅방 목록
    @GetMapping
    public ResponseEntity<BaseResponse<List<ChannelRes>>> getPersonalChatMessage(
        @RequestHeader("x-member-id") int memberId
    ) {
        List<ChannelRes> channelResList = channelService.getPersonalChatList(memberId);
        return BaseResponse.success(SuccessCode.SELECT_SUCCESS, channelResList);
    }

    //채널 정보 조회
    @GetMapping("/{channelId}")
    public ResponseEntity<BaseResponse<Channel>> getPersonalChat(
        @PathVariable String channelId,
        @RequestHeader("x-member-id") int memberId
    ) {
        //채팅방 조회
        Channel channel = channelService.readPersonalChat(channelId, memberId);
        return BaseResponse.success(SuccessCode.SELECT_SUCCESS, channel);
    }

    //고민상담소 퇴장
    @PutMapping("/exit/{consultId}")
    @Transactional
    public ResponseEntity<BaseResponse<String>> exitPersonalChat(
        @PathVariable int consultId,
        @RequestHeader("x-member-id") int sender //퇴장하는 사람
    ) throws IOException {

        //고민상담소 나가기
        consultService.exitConsultingRoom(consultId, sender);
        return BaseResponse.success(SuccessCode.DELETE_SUCCESS, "고민상담소를 퇴장하셨습니다.");
    }

    //고민상담소에서 강제로 내보내기
    @PutMapping("/expel/{consultId}")
    @Transactional
    public ResponseEntity<BaseResponse<String>> expelPersonalChat(
        @PathVariable int consultId,
        @RequestHeader("x-member-id") int sender //추방시키는 사람(방장인지 확인필요)
    ) {

        //강제로 추방시키기
        consultService.expelConsultingRoom(consultId, sender);
        return BaseResponse.success(SuccessCode.DELETE_SUCCESS, "참여자가 강제로 추방되었습니다.");
    }

    //공유된 고민 채팅 내역 조회
    @GetMapping("/shared/{channelId}")
    public ResponseEntity<BaseResponse<ChannelListRes>> getSharedChat(
        @PathVariable String channelId
//        @RequestHeader("x-member-id") int memberId
    ){
        ChannelListRes channelListRes = channelService.getSharedChat(channelId);
        return BaseResponse.success(SuccessCode.SELECT_SUCCESS, channelListRes);
    }
}
