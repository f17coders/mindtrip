package com.a303.missionms.global.client;

import com.a303.missionms.domain.member.dto.response.MemberBaseRes;
import com.a303.missionms.global.api.response.BaseResponse;
import java.util.List;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestParam;

@FeignClient(name = "notificationms")
public interface NotificationClient {

    @PostMapping("api/notificationms/v0/dailymission")
    BaseResponse<Integer> dailyMissionScheduling();

}